import cv2
import imutils


class SymbolDetector:
    def __init__(self, image, kernel_size):
        self.image = image
        self.resized_image = None
        self.dilated_image = None
        self.thresholded = None
        self.kernel_size = kernel_size
        self.ratio = None
        self.contours = None
        self.coordinates_list = []
        self.resize_image()
        self.process_image()

    def resize_image(self):
        read_image = cv2.imread(self.image)
        inverted_image = cv2.bitwise_not(read_image)  # invert the color if the background is white
        self.resized_image = imutils.resize(inverted_image, width=300)
        self.ratio = inverted_image.shape[0] / float(self.resized_image.shape[0])

    def process_image(self):
        # Dilate the contours on the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        self.dilated_image = cv2.dilate(self.resized_image, kernel)
        # convert the resized image to grayscale, blur it slightly, and threshold it
        gray = cv2.cvtColor(self.dilated_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.thresholded = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    def find_contours(self):
        # Find contours in the thresholded image
        cnts = cv2.findContours(self.thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find all the Countours
        self.contours = imutils.grab_contours(cnts)

    def compute_contour_centers(self):
        self.find_contours()
        for c in self.contours:
            # compute the center (cX, cY) of the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * self.ratio)
            cY = int((M["m01"] / M["m00"]) * self.ratio)
            object_coodinates = (c, cX, cY)
            self.coordinates_list.append(object_coodinates)

            # Sort the objects in the list from left to right, to detect the mathematical symbols form left to right
            def getKey(item):
                print(f"item = {item} \n\n key = {item[1]}")
                return item[1]
            self.coordinates_list = sorted(self.coordinates_list, key=getKey)
        return self.coordinates_list

    def recognize_symbol(self):
        pass

    def detect_symbols(self):
        self.__init__("formula.png", self.kernel_size)  # Reset the instance to get the last drawing on the image
        self.compute_contour_centers()
        self.display_contours_on_image()

    def display_contours_on_image(self):
        read_image = cv2.imread(self.image)
        for k in range(0, len(self.coordinates_list)):
            print(f" cX = {self.coordinates_list[k][1]}, cY = {self.coordinates_list[k][2]}\n")
            c = self.coordinates_list[k][0]

            # multiply the contour (x, y) coordinates by the resize ratio, then draw the contours
            c = c.astype("float")
            c *= self.ratio  # the counter is smaller if it is not multiplied by the ratio
            c = c.astype("int")
            cv2.drawContours(read_image, [c], -1, (0, 255, 0), 2)

            # display the name of the coontour on the image
            # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # show the output image
            cv2.imshow("Image", read_image)
            cv2.waitKey(0)


# image = cv2.imread("formula.png")
# sd = SymbolDetector(image="formula.png")
# sd.compute_contour_centers()
# sd.display_contours_on_image()

