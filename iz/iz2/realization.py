
import canny


canny.method_canny('../../images/test_512.jpg', 10, 3, 0.2, 0.4, 'sobel')
canny.method_canny('../../images/test_512.jpg', 20, 3, 0.1,0.8, 'previtta')