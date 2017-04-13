import tesserocr
import os


def batch_rec(img_path):
    # Recognize all the text that present in the path folder, and store the result in a txt time which
    # has the same name with the image
    if not os.path.exists(img_path):
        return

    dirs = os.listdir(img_path)
    for img in dirs:
        print("predicting "+img)
        try:
            content = tesserocr.file_to_text(img_path+img)
        except Exception:
            print("skip", img)
            continue

        # content = tesserocr.file_to_text(img_path+img)

        name, _ = img.split(".")
        with open(img_path + name + '_rec.txt', 'w') as file:
            file.write(content)


# print(tesserocr.tesseract_version())  # print tesseract-ocr version
# print(tesserocr.get_languages())  # prints tessdata path and list of available languages
#
# image = Image.open('sample.jpg')
# print(tesserocr.image_to_text(image))  # print ocr text from image
# # or
# print(tesserocr.file_to_text('sample.jpg'))

if __name__ == '__main__':
    batch_rec("recover_result/")
