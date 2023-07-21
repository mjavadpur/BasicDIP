import cv2

import heapq
import os
from collections import defaultdict

def compress_using_jpeg(image, quality):
    """
    Compresses the image using JPEG compression.
    :param quality: Compression quality (0-100), higher value means better quality.
    """
    encoded_image, compressed_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return compressed_image

def compress_using_png(image, compression_level):
    """
    Compresses the image using PNG compression.
    :param compression_level: Compression level (0-9), higher value means higher compression.
    """
    encoded_image, compressed_image = cv2.imencode('.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level])
    return compressed_image
def compress_using_huffman(IMAGE_PATH):
    
    h = HuffmanCoding(IMAGE_PATH)
    output_path = h.compress()
    return output_path
        
            

class HuffmanCoding:
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
    
    class HeapNode:
        
        def __init__(self, pixel_val, freq):
            self.pixel_val = pixel_val
            self.freq = freq
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.freq < other.freq
        
        def __eq__(self, other):
            if(other == None):
                return False
            if(not isinstance(other, self)):
                return False
            return self.freq == other.freq
    
    def make_frequency_dict(self, pixels):
        frequency = defaultdict(int)
        for pixel in pixels:
            frequency[pixel] += 1
        return frequency
    
    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)
    
    def merge_nodes(self):
        while(len(self.heap)>1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)
    
    def make_codes_helper(self, root, current_code):
        if(root == None):
            return
        if(root.pixel_val != None):
            self.codes[root.pixel_val] = current_code
            self.reverse_mapping[current_code] = root.pixel_val
            return
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")
    
    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)
    
    def get_encoded_image(self, pixels):
        encoded_image = ""
        for pixel in pixels:
            encoded_image += self.codes[pixel]
        return encoded_image
    
    def pad_encoded_image(self, encoded_image):
        extra_padding = 8 - len(encoded_image) % 8
        for i in range(extra_padding):
            encoded_image += "0"
        padded_info = "{0:08b}".format(extra_padding)
        padded_encoded_image = padded_info + encoded_image
        return padded_encoded_image
    
    def get_byte_array(self, padded_encoded_image):
        if(len(padded_encoded_image) % 8 != 0):
            print("Encoded image not padded properly")
            exit(0)
        b = bytearray()
        for i in range(0, len(padded_encoded_image), 8):
            byte = padded_encoded_image[i:i+8]
            b.append(int(byte, 2))
        return b
    
    def compress(self):
        filename, ext = os.path.splitext(self.image_path)
        output_path = filename + ".bin"
        with open(self.image_path, 'rb') as image_file, open(output_path, 'wb') as output:
            pixels = []
            byte = image_file.read(1)
            while(byte != b''):
                pixels.append(int.from_bytes(byte, byteorder='big'))
                byte = image_file.read(1)
            frequency = self.make_frequency_dict(pixels)
            self.make_heap(frequency)
            self.merge_nodes()
            self.make_codes()
            encoded_image = self.get_encoded_image(pixels)
            padded_encoded_image = self.pad_encoded_image(encoded_image)
            b = self.get_byte_array(padded_encoded_image)
            output.write(bytes(b))
        print("Compressed successfully!")
        return output_path
                        
            