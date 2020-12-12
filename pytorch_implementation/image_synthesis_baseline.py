import os
import model_loader
import cv2

def main():
    n = int(input("How many pictures do you want to produce: "))
    folder_name = input("Enter folder name: ")
    # generator_name = input("Which generator do you want to use")

    # load Generator
    model = model_loader.build(discriminator=False, encoder=False)

    # synthesize
    images = model_loader.synthesize(model[0],n,seed=0)

    # save stuff
    os.mkdir('img_syn_stylegan/' + folder_name)
    print(f'save')
    for i, img in enumerate(images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'img_syn_stylegan/{folder_name}/img{str(i).zfill(6)}.jpg',img_rgb)
    print(f'saved stuff')

if __name__ == '__main__':
    main()
