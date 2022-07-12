def visualize(self, masks, labels, img_paths, epoch, global_cam=None):
    select_id = [0, 5, 8, 13, 16, 21, 24, 29, 32, 37, 40, 45, 48, 53, 56, 61]  # [0,5,12,17,70,75,80,85]
    maskt, maskm = masks[0], masks[1]
    h, w = maskt.shape[2], maskt.shape[3]
    maskt = maskt.data.cpu().numpy()
    maskm = maskm.data.cpu().numpy()

    for id in select_id:

        mask = maskt[id].reshape(-1, h * w)
        mask = mask - np.min(mask, axis=1)
        mask = mask / np.max(mask, axis=1)
        # mask = 1 - mask
        mask = mask.reshape(h, w)
        # 归一化操作（最小的值为0，最大的为1）
        '''cam = weight[labels[id]:labels[id]+1].dot(global_cam[id].reshape((2048,h*w))).reshape(1,h * w)
        cam = cam - np.min(cam, axis=1)
        cam = cam / np.max(cam, axis=1)
        cam = cam.reshape(h, w)
        cam = np.uint8(255 * cam)'''
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * mask)
        # resize 图片尺寸与输入图片一致
        # 将图片和CAM拼接在一起展示定位结果结果
        img = cv2.imread(img_paths[id], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 384), cv2.INTER_LINEAR)
        # img = cv2.cvtColor(np.uint8(imgs[0][i]), cv2.COLOR_RGB2BGR)
        heightv, widthv, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (widthv, heightv)), cv2.COLORMAP_JET)
        # heatmap_cam = cv2.applyColorMap(cv2.resize(cam, (widthv, heightv)), cv2.COLORMAP_JET)
        # 生成热度图
        result = heatmap * 0.3 + img * 0.5
        # result_cam = heatmap_cam * 0.3 + img * 0.5

        path = './heatmap/' + str(epoch)
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(path + '/' + img_paths[id].split('/')[-2] + str(labels[id].data) + '_t' + str(id) + '.jpg', result)