import os

import imageio
def create_gif(img_path, gif_name, duration = 1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in os.listdir(img_path):
        temp = os.path.join(img_path, image_name)
        print(temp)
        frames.append(imageio.imread(temp))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def main():
    #这里放上自己所需要合成的图片文件夹路径
    image_path = r'/home/ryan/Desktop/ENeRF/debug/rendered_view/render__scannet_0113_duplicating_moving_toydesk2_duplicating_moving_test2'
    gif_name = 'demo3.gif'
    duration = 0.3        # 播放速度yuexiaoyuekuai
    create_gif(image_path, gif_name, duration)

if __name__ == '__main__':
    main()