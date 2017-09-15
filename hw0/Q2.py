import sys
from PIL import Image

filename = sys.argv[1]

orig_im = Image.open(filename)
orig_PixelAccess = orig_im.load()

new_im = Image.new(orig_im.mode, orig_im.size)
new_PixelAccess = new_im.load()

f= open('rgb.txt', 'w')

for i in range(new_im.size[0]):
	for j in range(new_im.size[1]):
		new_PixelAccess[i, j] = (((orig_PixelAccess[i, j])[0])//2, ((orig_PixelAccess[i, j])[1])//2, ((orig_PixelAccess[i, j])[2])//2)
#		f.write(str(((orig_PixelAccess[i, j])[0])//2) + ',' + str(((orig_PixelAccess[i, j])[1])//2) + ',' + str(((orig_PixelAccess[i, j])[2])//2) )
new_im.save('Q2.png')



