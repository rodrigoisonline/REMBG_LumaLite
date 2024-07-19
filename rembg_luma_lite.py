

import os
import glob
import numpy as np
from PIL import Image
from skimage import io
from time import sleep
import glob
import cv2


def bkg_cleaner(input_image_path, output_folder):

    def color_mask_pixel_removal1(image_in, image_out, target_color, tolerance=50):
        def is_within_tolerance(r, g, b, target_color, tolerance):
            tr, tg, tb = target_color
            return abs(r - tr) < tolerance and abs(g - tg) < tolerance and abs(b - tb) < tolerance
        def remove_target_color_pixels(img, target_color, tolerance):
            img = img.convert("RGBA")
            datas = img.getdata()
            newData = []
            for item in datas:
                r, g, b, a = item
                if is_within_tolerance(r, g, b, target_color, tolerance):
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            img.putdata(newData)
            return img
        img = Image.open(image_in)
        img = remove_target_color_pixels(img, target_color, tolerance)
        img.save(image_out, "PNG")

    import os
    import glob
    import numpy as np
    from PIL import Image
    from skimage import io
    import torch
    import torchvision.transforms as transforms
    from torch.autograd import Variable
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from saved_models.u2net import U2NET   # Adjust the import path as per your directory structure
    from time import sleep
    import glob
    import cv2

    def normPRED(d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn


    def blur_process_images(image_dir, output_dir, blur_ksize=(5, 5), sigmaX=0):
        #for filename in os.listdir(image_dir):
        #    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_path = image_dir
        output_path = output_dir
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            black_background = np.zeros_like(bgr)
            blurred_alpha = cv2.GaussianBlur(alpha, blur_ksize, sigmaX)
            smoothed_image = np.concatenate((black_background, blurred_alpha[:, :, np.newaxis]), axis=2)
            cv2.imwrite(output_path, smoothed_image)

    def clean_white_pixels(image_path):
        img = Image.open(image_path)
        # Convert the image to RGB mode (if not already)
        img = img.convert("RGB")
        # Get the image dimensions
        width, height = img.size
        # Loop through each pixel
        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                r, g, b = img.getpixel((x, y))          
                # If the pixel is not white (r, g, b != 255), turn it black
                if r != 255 or g != 255 or b != 255:
                    img.putpixel((x, y), (0, 0, 0))
        return img

    def c_clean_white_pixels(image_path):
        img = Image.open(image_path)
        # Convert the image to RGBA mode (if not already)
        img = img.convert("RGBA")
        # Get the image dimensions
        width, height = img.size
        # Loop through each pixel
        for x in range(width):
            for y in range(height):
                # Get the RGBA values of the pixel
                r, g, b, a = img.getpixel((x, y))          
                # If the pixel is not white (r, g, b != 255), set it to alpha-transparent white
                if r != 0 or g != 0 or b != 0:
                    img.putpixel((x, y), (0, 0, 0, 0))  # Alpha is set to 0 for transparency
        return img

    def save_output(image_name, pred, d_dir):
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = Image.fromarray(predict_np * 255).convert('RGB')
        img_name = image_name.split("/")[-1]
        image = io.imread(image_name)
        imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        pb_np = np.array(imo)
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        imo.save(os.path.join(d_dir, imidx + '.png'))

    def bkpredict_u2net(image_path, model, output_dir):
        image = Image.open(image_path)
        image = transforms.Resize((320, 320))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image[None,:,:,:])
        model.eval()
        with torch.no_grad():
            predictions = model(image)
            prediction = predictions[0]  # Access the main output
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze().cpu()


    def predict_u2net(image_path, model, output_dir):
        from time import sleep
        print("image path = "+ image_path)
        #sleep(345678)
        image = Image.open(image_path)
        image = transforms.Resize((320, 320))(image)
        image = transforms.ToTensor()(image)
        #factorymask 
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image[None,:,:,:])
        #halphligthmask image = transforms.Normalize((0.485, 0.456, 0.406), (0.157, 0.118, 0.1))(image[None,:,:,:])
        #FacialMask image = transforms.Normalize((0.485, 0.456, 0.406), (0.256, 0.218, 0.1))(image[None,:,:,:])
        #image = transforms.Normalize((0.485, 0.456, 0.406), (0.200, 0.180, 0.180))(image[None,:,:,:])
        model.eval()
        with torch.no_grad():
            predictions = model(image)
            prediction = predictions[0]  # Access the main output
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze().cpu()
        save_output(image_path, prediction, output_dir)

    if __name__ == '__main__':

        #if not os.path.exists("test_data/mask_results"):
        #    os.makedirs("test_data/mask_results")
            
        model_path = 'saved_models/weights/u2net.pth'
        net = U2NET(3, 1) 
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load model on CPU
        net.eval()
    
    predict_u2net(input_image_path, net, output_folder)
    file_name= input_image_path.replace("face_pool/id_left/","") #--------------------------------------------------PRECISA MODIFICAR 
    output_file_dir=(output_folder+file_name)
    color_mask_pixel_removal1(output_folder+file_name,output_file_dir, (186, 186, 186), tolerance=15)
    processed_img = clean_white_pixels(output_file_dir)
    processed_img.save(output_file_dir)
    processed_img = c_clean_white_pixels(output_file_dir)
    processed_img.save(output_file_dir)

    blur_ksize = (15, 15)
    sigmaX = 0.9
    blur_process_images(output_file_dir, output_file_dir, blur_ksize, sigmaX)
    return output_file_dir
    # - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - - 

def img_process(image, dir_output):
    b = bkg_cleaner(image, dir_output)
    print(b)
    a_image = Image.open(b).convert('RGBA')   # = mask
    b_image = Image.open(image).convert('RGBA') # = image

    #convert images to numpy array(convert to numbers, 
    #beside is hard to human understand, it's easy and faster to computer)
    a_array = np.array(a_image)
    a_array_2 = Image.fromarray(a_array.astype(np.uint8))
    b_array = np.array(b_image)
    if a_image.size != b_image.size:
        raise ValueError("Images must have the same dimensions")
    # Compute the mask by subtracting b from a
    mask_array = np.subtract(b_array, a_array) # SE INVERTER VIRA SILLUETA 
    mask_image = Image.fromarray(mask_array.astype(np.uint8))
    #save_directory = os.path.join(dir_output)
    mask_image.save(b)

img_process("face_pool/id_left/imageoutput1.png", "files/temp/")













