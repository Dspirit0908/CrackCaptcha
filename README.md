# CrackCaptcha
A Simple Framework to Crack Captcha Based on Tensorflow

## How to use
The Easiest Way:
1. Put all your labeled images into a folder
2. Instantiate the CrackCaptcha class: `CrackCaptcha = CrackCaptcha(your_images_folder)`
3. Call auto_crack to train a model: `CrackCaptcha.auto_crack()`  
If the number of your labeled images is more than 2500, you can use: `CrackCaptcha.auto_crack(fine_tuning=False)`
4. Then you can use `CrackCaptcha.crack_one_image(image_file)` to return the predict text. **Good Luck!**