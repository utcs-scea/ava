/***********************************************************************
 *
 * Copyright (c) 2017-2019 Gyrfalcon Technology Inc. All rights reserved.
 * See LICENSE file in the project root for full license information.
 *
 ************************************************************************/

/**
 * \file GTILib.h
 * \brief GTI header file includes the public functions of GTI SDK library.
 */
#ifndef GYRFALCONTECH_GTILIB_H
#define GYRFALCONTECH_GTILIB_H

#ifdef GTISDK_DLL_EXPORT
#define GTISDK_API __declspec(dllexport)
#else
#define GTISDK_API
#endif

enum GTI_DEVICE_STATUS {
  GTI_DEVICE_STATUS_ERROR,
  GTI_DEVICE_STATUS_ADDED,
  GTI_DEVICE_STATUS_REMOVED,
  GTI_DEVICE_STATUS_IDLE,
  GTI_DEVICE_STATUS_LOCKED,
  GTI_DEVICE_STATUS_RUNNING

};
enum GTI_DEVICE_TYPE {
  GTI_DEVICE_TYPE_ALL,
  GTI_DEVICE_USB_FTDI,
  GTI_DEVICE_USB_EUSB,
  GTI_DEVICE_PCIE,
  GTI_DEVICE_VIRTUAL,
  GTI_DEVICE_USB_NATIVE,
  GTI_DEVICE_TYPE_BUT
};

enum GTI_CHIP_MODE {
  FC_MODE,
  LEARN_MODE,
  SINGLE_MODE,
  SUBLAST_MODE,
  LASTMAJOR_MODE,
  LAST7x7OUT_MODE,
  SUM7x7OUT_MODE,
  GTI_CHIP_MODE_BUT
};

enum TENSOR_FORMAT {
  TENSOR_FORMAT_BINARY,  // default is char
  TENSOR_FORMAT_BINARY_INTEGER,
  TENSOR_FORMAT_BINARY_FLOAT,
  TENSOR_FORMAT_TEXT,
  TENSOR_FORMAT_JSON,
  TENSOR_FORMAT_BUT,
};

class GtiDevice;
class GtiContext;
class GtiModel;
class GtiTensor {
 public:
  int width;
  int height;
  int depth;
  int stride;
  void *buffer;
  int size;  // buffer size;
  TENSOR_FORMAT format;
  void *tag;  // customer tag. will be copied from the input tensor to output tensors.

  void *next;        // GTI reserved for future use.
  void *callbackFn;  // GTI internal use, don't touch.
};

extern "C" {
typedef void (*GtiDeviceNotificationCallBack)(GtiDevice *device);
typedef void (*GtiEvaluateCallBack)(GtiModel *model, const char *layerName, GtiTensor *output);

/*!
    \fn GTISDK_API GtiModel *GtiCreateModel(const char *modelFile);
    \brief Create a GTI Model object from a Model File
    \param[in] modelFile -- A constant char string containing the name of the model file
    \return -- A pointer to the created GtiModel object
*/
GTISDK_API GtiModel *GtiCreateModel(const char *modelFile);

/*!
        \fn GTISDK_API GtiModel *GtiCreateModelFromBuffer(void *modelBuffer, int size);
        \brief Create a GTI Model object given a buffer that contains the entire model
        \param[in] modelBuffer -- A pointer to a buffer containing a model as a byte array
        \param[in] modelSize -- An integer containing the model's size in bytes
        \return -- A pointer to the created GtiModel object
*/
GTISDK_API GtiModel *GtiCreateModelFromBuffer(void *modelBuffer, int size);

/*!
        \fn GTISDK_API int GtiDestroyModel(GtiModel *model);
        \brief Destory a GTI Model object
        \param[in] model -- A pointer to a GtiModel object
        \return -- 1 on success
*/
GTISDK_API int GtiDestroyModel(GtiModel *model);

/*!
        \fn GTISDK_API int GtiEvaluateWithCallback(GtiModel *model,GtiTensor * input, GtiEvaluateCallBack fn);
        \brief Evaluate an input GtiTensor object with a callback function
        This function sends input data as a GtiTensor object to the established model.  The model
        filters the data through all layers in the network, and call the callback function after
        the last layer
        \param[in] model -- A pointer to the model object
        \param[in] input -- A pointer to the input GtiTensor object
        \param[in] fn    -- A pointer to the callback function
        \return -- always 1.
*/
GTISDK_API int GtiEvaluateWithCallback(GtiModel *model, GtiTensor *input, GtiEvaluateCallBack fn);

/*!
        \fn GTISDK_API GtiTensor * GtiEvaluate(GtiModel *model,GtiTensor * input);
        \brief Evaluate an input GtiTensor object
        This function sends input data as a GtiTensor object to the established model.  The model
        filters the data through all layers in the network, and output the results as a GtiTensor
        object.
        \param[in] model -- A pointer to the model object
        \param[in] input -- A pointer to the input GtiTensor object
        \return -- A pointer to the output GtiTensor object
*/
GTISDK_API GtiTensor *GtiEvaluate(GtiModel *model, GtiTensor *input);

/*!
        \fn GTISDK_API const char * GtiImageEvaluate(GtiModel *model, const char *image, int height, int width, int
   depth); \brief Evaluate an input image object presented as an array of bytes This function sends input data as an
   image arranged in an array of bytes to the established model.  The model filters the data through all layers in the
   network, and output the results as an array of bytes. \param[in] model -- a pointer to a model object \param[in]
   image -- a char string, the input image file \param[in] height -- height of image in number of pixels \param[in]
   width -- width of image in number of pixels \param[in] depth -- depth of image in number of channels per pixel
        \return -- a pointer to the output data as an array of bytes
*/
GTISDK_API const char *GtiImageEvaluate(GtiModel *model, const char *image, int height, int width, int depth);

/*!
        \fn GTISDK_API void GtiDecomposeModelFile(const char *modelFile);
        \brief Decompose a model file into per layer model defintion files
        A model file represents a complete network that consists of multiple layers.
        This function breaks the model file into multiple individual files
        each contains definition of a layer. A separate json file that
        contains the decomposed structure of the network is also generated.
        \param[in] modelFile -- A constant char string containing the name of a model file
        \return None
*/
GTISDK_API void GtiDecomposeModelFile(const char *modelFile);

/*!
        \fn GTISDK_API void GtiComposeModelFile(const char *jsonFile, const char *modelFile);
        \brief Compose individual layers into a model file.
        Given a network structure definition defined with a json file, this function assembles individual
        layer definitions into a complete model file.  Each layer is stored in a file with its filename
        recorded in the json file.  The output is a new model file that consists of all layers.
        \param[in] jsonFile -- A constant char string for the name of the json file that contains the network structure
        \param[in] modelFile -- A constant char string containing the name of a model file
        \return -- None
*/
GTISDK_API void GtiComposeModelFile(const char *jsonFile, const char *modelFile);

/*!
        \fn GTISDK_API const char * GtiGetSDKVersion();
        \brief This function returns GTISDK version as a string
        \return -- GTISDK version
*/
GTISDK_API const char *GtiGetSDKVersion();

/*!
        \fn GTISDK_API GtiContext *GtiGetContext();
        \brief This function returns a point to the current GtiContext object
        \return -- GTI Context object
*/
GTISDK_API GtiContext *GtiGetContext();

/*!
        \fn GTISDK_API void GtiRegisterDeviceEventCallBack(GtiContext * context, GtiDeviceNotificationCallBack fn);
        \brief This function registers an event callback function of type void func(GtiDevice * device)
        \param[in,out] context -- The GtiContext to register event with
        \param[in] fn -- A function pointer to an event callback function of type void func(GtiDevice * device)
        \return -- None
*/
GTISDK_API void GtiRegisterDeviceEventCallBack(GtiContext *context, GtiDeviceNotificationCallBack fn);

/*!
        \fn GtiRemoveDeviceEventCallBack(GtiContext * context, GtiDeviceNotificationCallBack fn);
        \brief This function unregisters an event callback function of type void func(GtiDevice * device)
        \param[in,out] context -- The GtiContext to unregister event with
        \param[in] fn -- Pointer to the event callback function
        \return -- None
*/
GTISDK_API void GtiRemoveDeviceEventCallBack(GtiContext *context, GtiDeviceNotificationCallBack fn);

/*!
        \fn GTISDK_API GtiDevice *GtiGetDevice(GtiContext * context, const char *devicePlatformName);
        \brief Get an availble GtiDevice object that have required platform name
        \param[in] context -- The GtiContext object that contains the GtiDevice
        \param[in] devicePlatformName -- A string represeting the platform's name of a device
        \return -- A pointer to the available GtiDevice object
*/
GTISDK_API GtiDevice *GtiGetDevice(GtiContext *context, const char *devicePlatformName);
/*!
        \fn GTISDK_API GtiDevice *GtiGetAvailableDevice(GtiContext * context, GTI_DEVICE_TYPE deviceType);
        \brief Get an avilable GtiDevice object from the list of unused devices that satisfies the required type.
        \param[in] context -- The GtiContext object that has the GtiDevice
        \param[in] deviceType -- Enumerated device type
        \return -- A pointer to the available GtiDevice object
*/
GTISDK_API GtiDevice *GtiGetAvailableDevice(GtiContext *context, GTI_DEVICE_TYPE deviceType);

// GTISDK_API int GtiLockDevice(GtiDevice *device);  // not implemented

/*!
        \fn GTISDK_API int GtiDeviceRead(GtiDevice *device,unsigned char * buffer, unsigned int length);
        \brief read data from the device.
        \param[in] device -- A pointer to the GtiDevice
        \param[in] buffer -- A pointer to the read buffer
        \param[in] length -- read buffer length
        \return -- the number of bytes read, minus numbers mean errors
*/
GTISDK_API int GtiDeviceRead(GtiDevice *device, unsigned char *buffer, unsigned int length);

/*!
        \fn GTISDK_API int GtiDeviceWrite(GtiDevice *device,unsigned char * buffer, unsigned int length);
        \brief write data to the device.
        \param[in] device -- A pointer to the GtiDevice
        \param[in] buffer -- A pointer to the write buffer
        \param[in] length -- write buffer length
        \return -- the number of bytes wrote, minus numbers mean errors
*/
GTISDK_API int GtiDeviceWrite(GtiDevice *device, unsigned char *buffer, unsigned int length);

/*!
        \fn GTISDK_API int GtiUnlockDevice(GtiDevice *device);
        \brief Release the device and set the device state free.
        \param[in] device -- A pointer to the GtiDevice to be unlocked
        \return -- True, if the device is successfully unlocked; False, if the device cannot be unlocked
*/
GTISDK_API int GtiUnlockDevice(GtiDevice *device);

/*!
        \fn GTISDK_API int GtiResetDevice(GtiDevice *device);
        \brief Try to reset the device
        \param[in] device -- A pointer to the GtiDevice
        \return -- True, if the device is successfully reset; False, if the device is not reset
*/
GTISDK_API int GtiResetDevice(GtiDevice *device);

/*!
        \fn GTISDK_API GTI_DEVICE_TYPE GtiGetDeviceType(GtiDevice *device);
        \brief Get the given device's type definition.
        \param[in] device -- A pointer to the GtiDevice object
        \return -- Enumerated device type
*/
GTISDK_API GTI_DEVICE_TYPE GtiGetDeviceType(GtiDevice *device);

/*!
        \fn GTISDK_API const char *GtiGetDeviceName(GtiDevice *device);
        \brief Get the name of the device in a string
        \param[in] device -- A pointer to the GtiDevice object
        \return -- A null terminalted string containing the device's name
*/
GTISDK_API const char *GtiGetDeviceName(GtiDevice *device);

/*!
        \fn GTISDK_API const char *GtiGetDevicePlatformName(GtiDevice *device);
        \brief Get the device's platform name in a string
        \param[in] device -- A pointer to the GtiDevice
        \return -- A null terminalted string representing the platform name
*/
GTISDK_API const char *GtiGetDevicePlatformName(GtiDevice *device);

/*!
        \fn GTISDK_API GTI_DEVICE_STATUS GtiCheckDeviceStatus(GtiDevice *device);
        \brief Get the device's status
        \param[in] device -- A pointer to the GtiDevice object
        \return -- Enumerated device status
*/
GTISDK_API GTI_DEVICE_STATUS GtiCheckDeviceStatus(GtiDevice *device);

/*!
        \fn GTISDK_API int GtiLoadModel(GtiDevice *device, const char * modelUrl, GTI_CHIP_MODE mode, int networkId);
        \brief Load a model to a device directly
        \param[in] device -- A pointer to the GtiDevice object
        \param[in] modelUrl -- URL path of the model file in a null terminated string
        \param[in] mode -- chip mode, eusb, ftdi, or others
        \param[in] networkId -- ID of the network
        \return -- True, if the model is loaded successfully; False, if the model cannot be loaded
*/
GTISDK_API int GtiLoadModel(GtiDevice *device, const char *modelUrl, GTI_CHIP_MODE mode, int networkId);
/*!
        \fn GTISDK_API int GtiChangeModelMode(GtiDevice *device, GTI_CHIP_MODE mode);
        \brief Change the mode of the device's loaded model
        \param[in] device -- A pointer to the GtiDevice object
        \param[in] mode -- device's mode (chip mode), e.g. eusb, ftdi, or others
        \return -- True, if the mode is changed successfully; False otherwise
*/
GTISDK_API int GtiChangeModelMode(GtiDevice *device, GTI_CHIP_MODE mode);
/*!
        \fn GTISDK_API int GtiChangeModelNetworkId(GtiDevice *device, int networkId);
        \brief Change the network ID of the device's current model
        \param[in] device -- A pointer to the GtiDevice object
        \param[in] networkId -- ID of the network
        \return -- True, if the network ID is changed successfully; False otherwise
*/
GTISDK_API int GtiChangeModelNetworkId(GtiDevice *device, int networkId);

/*!
        \fn int GtiHandleOneFrame(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int InputLen,
                                  unsigned char *OutputBuffer,unsigned int OutputLen)
        \brief This function sends a 224x224x3 byte image data buffer to GTI chip and extracts
        the result in fixed point 32-bit data format.
        Image data buffer contains an image in 8 bit per channel plannar RGB or BGR depending on model.
        Usually models trained by Caffe is BGR, otherwise is RGB.
        \param[in] Device -- A pointer to the device object created by function GtiDeviceCreate
        \param[in] Image224Buffer -- The image data buffer, a 224x224x3 byte array
        \param[in] InputLen -- The length in bytes of the image data buffer Image224Buffer
        \param[out] OutputBuffer -- Pointer to a prepared buffer for the output data
        \param[in] OutputLen -- Length of the output data in number of output elements
        \return -- 1 on succee; 0 on fail
*/
GTISDK_API int GtiHandleOneFrame(GtiDevice *device, unsigned char *inputBuffer, unsigned int inputLen,
                                 unsigned char *outputBuffer, unsigned int outLen);

/*!
        \fn int GtiHandleOneFrameFloat(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int InputLen,
                                       float *OutputBuffer, unsigned int OutputLen)
        \brief This function sends a 224x224x3 byte image data buffer to GTI chip and extracts the result
        in floating point 32-bit data format.
        Image data buffer contains an image in 8 bit per channel plannar RGB or BGR depending on model.
        Usually models trained by Caffe is BGR, otherwise is RGB.
        \param[in] Device -- A pointer to the device object created by function GtiDeviceCreate
        \param[in] Image224Buffer -- The image data buffer, a 224x224x3 byte array
        \param[in] InputLen -- The length in bytes of the image data buffer Image224Buffer
        \param[out] OutputBuffer -- Pointer to a prepared buffer for the output data
        \param[in] OutputLen -- Length of the output data in number of output elements
        \return -- 1 on succee; 0 on fail
*/
GTISDK_API int GtiHandleOneFrameFloat(GtiDevice *device, unsigned char *inputBuffer, unsigned int inputLen,
                                      float *outputBuffer, unsigned int outLen);

/*!
        \fn unsigned int GtiGetOutputLength(GtiDevice *Device)
        \brief This function returns a device's output data length
        \param[in] Device -- A pointer to the device object
        \return -- Output data length in number of elements
*/
GTISDK_API unsigned int GtiGetOutputLength(GtiDevice *device);

/*!
        \fn GTISDK_API GtiDevice *GtiDeviceCreate(int DeviceType, char *FilterFileName, char *ConfigFileName);
        \brief Create a device object that represents a hardware device chip
        This function sets up which port is used to connect to a device, chooses which Gnet type
        and CNN mode to be used, it also allocates memory for internal use.
        The created device is of mode "FC_MODE" if mode is not defined by environment variable "GTI_CHIP_MODE"
        Filter file is GTI defined .dat coefficient file, for example: gnet32_fc128_20class.bin.
        Config file is GTI defined ASCII file for storing configuration information, for example: userinput.txt.
        These files can be found under the data folder.
        \param[in] DeviceType -- 0=FTDI; 1=EUSB;  2=PCIe
        \param[in] FilterFileName -- GTI defined .dat coef file to config network
        \param[in] ConfigFileName -- GTI defined ASCII file to config GTISDK usage
        \return -- A point to the created GtiDevice object

        \obsolete -- This function is obsolete. Support will be removed
*/
GTISDK_API GtiDevice *GtiDeviceCreate(int DeviceType, char *FilterFileName, char *ConfigFileName);

/*!
        \fn GTISDK_API void GtiDeviceRelease(GtiDevice *Device);
        \brief This function releases GTI chip and memory resource.
        \param[in] Device -- A point to the GtiDevice object to be released
        \return -- None

        \obsolete -- This function is obsolete. Support will be removed
*/
GTISDK_API void GtiDeviceRelease(GtiDevice *Device);

/*!
        \fn GTISDK_API int GtiOpenDevice(GtiDevice *Device, char *DeviceName);
        \brief Open the device for parameter setting and data transferring
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \param[in] DeviceName -- The name of the hardware device's device handle, e.g. /dev/sg2
        \return -- Always 1

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiOpenDevice(GtiDevice *Device, char *DeviceName);

/*!
        \fn void GtiCloseDevice(GtiDevice *Device)
        \brief Close the device.
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \return -- None.

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API void GtiCloseDevice(GtiDevice *Device);

/*!
        \fn void GtiSelectNetwork(GtiDevice *Device, int NetworkId)
        \brief Select which one to use in a device.
        This function selects which one to use if multiple networks are loaded. The network Id
        is the sequence (order) number of network defined in the Network structure file. The NetworkId
        number starts from 0.  In the file, NetworkId is 0, 1, 2, 3, etc.
        \param[in] Device -- The device created by GtiDeviceCreate
        \param[in] NetworkId -- The order number of network defined in network structure file
        \return -- None

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API void GtiSelectNetwork(GtiDevice *Device, int NetworkId);

/*!
        \fn int GtiInitialization(GtiDevice *Device)
        \brief Initializes a devices running environment
        This function initializes device's SDK library environment. It loads CNN co-efficient filter data to GTI chip.
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \return -- Always 1

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiInitialization(GtiDevice *Device);

/*!
        \fn int GtiSendImage(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int BufferLen)
        \brief Send an image data buffer to a device with data in fixed point 8-bit format.
        This function sends a 224x224x3 image data buffer to a device (e.g. a GTI chip).
        Image data buffer contains an image in 8 bit per channel plannar RGB or BGR depending on model.
        Usually models trained by Caffe is BGR, otherwise is RGB.
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \param[in] Image224Buffer -- A pointer to the 224x224x3 fixed point image data buffer
        \param[in] BufferLen -- Number of elements in the input buffer
        \return -- Always 0.

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiSendImage(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int BufferLen);

/*!
        \fn int GtiSendImageFloat(GtiDevice *Device, float *Image224Buffer, unsigned int BufferLen)
        \brief Send an image data buffer to a device with data in floating point format
        This function sends a 224x224x3 image data buffer to a device (e.g. a GTI chip).
        Image data buffer contains an image in 32 bit per channel plannar RGB or BGR depending on model.
        Usually models trained by Caffe is BGR, otherwise is RGB.
        In Image224Buffer, each element size is 4 bytes, i.e. sizeof(float).
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \param[in] Image224Buffer -- A pointer to the 224x224x3 floating point image data buffer
        \param[in] BufferLen -- Number of elements in the input buffer
        \return -- Always 0.

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiSendImageFloat(GtiDevice *Device, float *Image224Buffer, unsigned int BufferLen);

/*!
        \fn int GtiSendTiledImage(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int BufferLen)
        \brief Send a tiled image data buffer to a device
        This function sends a 224x224x3 tiled image data buffer to a device (e.g. a GTI chip).
        Image data buffer contains an image in 8 bit per channel plannar RGB or BGR depending on model.
        Usually models trained by Caffe is BGR, otherwise is RGB.
        In Image224Buffer, each channel's pixel order is tile scanned.
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \param[in] Image224Buffer -- A pointer to the 224x224x3 byte tile image data buffer
        \param[in] BufferLen -- Number of elements in the input buffer
        \return -- Always 0.

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiSendTiledImage(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int BufferLen);

/*!
        \fn int GtiGetOutputData(GtiDevice *Device, unsigned char *OutputBuffer, unsigned int BufferLen)
        \brief Get the output data from a device as fixed point data
        This function gets output data from a device (e.g. a GTI chip). The output data is 8-bit integer.
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \param[out] OutputBuffer -- A buffer to store the output data in fixed point format
        \param[in] BufferLen -- Number of elements in the output buffer
        \return -- Always 0

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiGetOutputData(GtiDevice *Device, unsigned char *OutputBuffer, unsigned int BufferLen);

/*!
        \fn int GtiGetOutputDataFloat(GtiDevice *Device, float *OutputBuffer, unsigned int BufferLen)
        \brief Get the output data from a device as floating point data
        This function gets output data from the device (e.g. a GTI chip).
        The output data is in floating point format.
        In OutputBuffer each element's size is 4 bytes, i.e. sizeof(float).
        \param[in] Device -- A pointer to the device object created by GtiDeviceCreate
        \param[out] OutputBuffer -- A buffer to store the output data in float point format
        \param[in] BufferLen -- Number of elements in the output buffer
        \return -- Always 0

        \obsolete This function is obsolete. Support will be removed
*/
GTISDK_API int GtiGetOutputDataFloat(GtiDevice *Device, float *OutputBuffer, unsigned int BufferLen);
}

#endif /* ifndef GYRFALCONTECH_GTILIB_H */
