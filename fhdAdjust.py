# read an ilumination pattern from file
# read and fhd file and multiply every layer with the pattern
# write out the result

from PIL import Image, ImageMath
from cStringIO import StringIO as StrIO
import re


input_path = "C:\\tmp\\_bethannie_hollow.fhd"
output_path = "C:\\tmp\\_bethannie_hollow_adjusted.fhd"
illum_path = "Q:\\Projects\\scripts\\python\\sparkmaker\\img\\lumamap_blur.png"

illum_pattern = Image.open(illum_path)

blk = Image.new("L", (1920,1080), color=0)

out_file = open(output_path, 'wb')

with open(input_path, "rb") as infile:

    datasize = -1
    linenum = 0
    while True:

        line = infile.readline()
        if not line:
            break


        if line.startswith(";dataSize:"):
            datasize = int(re.search("dataSize:[0-9]+", line).group()[9:])
            # remove it from the stream
            line = ""

        elif line.startswith("{{"):  # image data
            if datasize == -1:
                print "image found but no data size determined"
                continue

            img = infile.read(datasize)
            tail = infile.read(4)
            if tail != "\n}}\n":
                print " - meh   !!!!"
                print list(bytes(tail))
                break
            else:
                #print " - ok"
                png_in = StrIO(img)
                image_in = Image.open(png_in).copy()
                png_in.close()

                #multiply image with the pattern
                img_adj = Image.composite(image_in, blk, illum_pattern)

                outimg = StrIO()
                img_adj.save(outimg, "PNG", dpi=(144, 144))
                imgdata = outimg.getvalue()


                if len(imgdata) < 100000 and linenum > 3:  # FHD 100k bug
                    line = ";dataSize:%d\n{{\n"%len(imgdata) + imgdata + "\n}}\n"
                else:
                    line = ";dataSize:%d\n{{\n" % len(img) + img + "\n}}\n"
                outimg.close()

        elif line.startswith(";L:"):  # comments
            datasize = -1
            linenum = int(re.search("L:[0-9]+", line).group()[2:])

        else:
            pass

        # pipe to output
        if line:
            out_file.write(line)

        if linenum % 20 == 0:
            print "\r", 'processed line', linenum,

    infile.close()

out_file.close()
print "done."