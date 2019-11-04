#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <iostream>
#include "FreeImage.h"

using namespace std;

void writeRGBImageToFile(char* fileName, unsigned char* bytes, int w, int h)
{
	FreeImage_Initialise();

	FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(fileName);
	FIBITMAP* image = FreeImage_ConvertFromRawBits((BYTE*)bytes, w,
		h, w * 4, 32,
		0xFF000000, 0x00FF0000, 0x0000FF00);
	FreeImage_Save(format, image, fileName);

	FreeImage_Unload(image);

	FreeImage_DeInitialise();

	return;
}

void writeGrayImageToFile(char* fileName, unsigned char* bytes, int w, int h)
{
	FreeImage_Initialise();

	FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(fileName);
	FIBITMAP* image = FreeImage_ConvertFromRawBits((BYTE*)bytes, w,
		h, w, 32,
		0xFF, 0xFF, 0xFF);
	FreeImage_Save(format, image, fileName);

	FreeImage_Unload(image);

	FreeImage_DeInitialise();

	return;
}



void readRGBImageFromFile(char* fileName, unsigned char*& bytes, int& w, int& h)
{
	FreeImage_Initialise();

	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileName, 0);
	FIBITMAP* image = FreeImage_Load(format, fileName);

	FIBITMAP* temp = image;
	image = FreeImage_ConvertTo32Bits(image);
	FreeImage_Unload(temp);

	unsigned int pitch = FreeImage_GetPitch(image);
	w = FreeImage_GetWidth(image);
	h = FreeImage_GetHeight(image);
	int lineWidth = w * 4;

	bytes = new unsigned char[h * w * 4];

	for (int i = 0; i < h; i++)
	{
		unsigned char* src = FreeImage_GetScanLine(image, i);
		unsigned char* dst = bytes + i * lineWidth;

		memcpy(dst, src, lineWidth);
	}

	FreeImage_Unload(image);

	FreeImage_DeInitialise();
}

#endif