def laplassian(path, standard_deviation, kernel_size, lower_bound, upper_bound, operator):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    laplas_filter = [[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]]

    tresh = utils.convolution(img, Gx)
    cv2.threshold(
        laplacian_img, bound, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(tresh)
