// clang-format off
ava_name("Gyrfalcon Technology Plai SDK");
ava_version("4.4.0.3-2803");
ava_identifier(GTI);
ava_number(8);
ava_libs(-L/home/hyu/Downloads/GTISDK-Linux_x86_64_v4.4.0.3/Lib/Linux/x86_64 -lGTILibrary-static -lgomp -lftd3xx-static);
ava_cxxflags(-std=c++11);
ava_export_qualifier(GTISDK_API);
ava_soname(GTILibrary-static.so);
// clang-format on

ava_non_transferable_types { ava_handle; }

#include <GTILib.h>

ava_type(GtiContext *) { ava_handle; }
ava_type(GtiDevice *) { ava_handle; }
ava_type(GtiModel *) { ava_handle; };

ava_type(GtiTensor) {
  GtiTensor *ava_self;
  ava_field(width);
  ava_field(height);
  ava_field(depth);
  ava_field(stride);
  ava_field(buffer) {
    ava_buffer(ava_self->size);
    ava_lifetime_coupled(ava_self);
  }
  ava_field(size);
  ava_field(format);
  ava_field(tag) { ava_opaque; }
  ava_field(next) { ava_opaque; }
  ava_field(callbackFn) { ava_opaque; } /* ava_callback, void* */;
}

// ava_callback_decl void _GtiDeviceNotificationCallBack(GtiDevice *device);
// ava_callback_decl void _GtiEvaluateCallBack(GtiModel *model,const char *layerName, GtiTensor * output) {
//    ava_argument(layerName) {
//        ava_buffer(strlen(layerName) + 1);
//        ava_in;
//    }
//
//    ava_argument(output) {
//        ava_buffer(1); ava_out;
//    }
//}

GtiModel *GtiCreateModel(const char *modelFile) {
  ava_argument(modelFile) {
    ava_buffer(strlen(modelFile) + 1);
    ava_in;
  }

  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

GtiModel *GtiCreateModelFromBuffer(void *modelBuffer, int size) {
  ava_argument(modelBuffer) {
    ava_buffer(size);
    ava_in;
  }

  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

int GtiDestroyModel(GtiModel *model) { ava_argument(model) ava_deallocates; }

// int GtiEvaluateWithCallback(GtiModel *model, GtiTensor * input, GtiEvaluateCallBack fn) {
//    ava_argument(input) {
//        ava_buffer(1); ava_in;
//    }
//
//    ava_argument(fn) {
//        ava_callback(_GtiEvaluateCallBack);
//    }
//}

GtiTensor *GtiEvaluate(GtiModel *model, GtiTensor *input) {
  ava_argument(input) {
    ava_buffer(1);
    ava_in;
    ava_out;
  }

  ava_return_value {
    ava_buffer(1);
    ava_out;
    ava_lifetime_manual;
  }
}

const char *GtiImageEvaluate(GtiModel *model, const char *image, int height, int width, int depth) {
  ava_argument(image) {
    ava_buffer(height * width * depth);
    ava_in;
  }
}

void GtiDecomposeModelFile(const char *modelFile) {
  ava_argument(modelFile) {
    ava_buffer(strlen(modelFile) + 1);
    ava_in;
  }
}

void GtiComposeModelFile(const char *jsonFile, const char *modelFile) {
  ava_argument(jsonFile) {
    ava_buffer(strlen(jsonFile) + 1);
    ava_in;
  }

  ava_argument(modelFile) {
    ava_buffer(strlen(modelFile) + 1);
    ava_in;
  }
}

const char *GtiGetSDKVersion() {
  const char *ret;
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
  }
}

GtiContext *GtiGetContext();

// void GtiRegisterDeviceEventCallBack(GtiContext * context, GtiDeviceNotificationCallBack fn) {
//    ava_argument(fn) {
//        ava_callback(_GtiDeviceNotificationCallBack);
//    }
//}
//
// void GtiRemoveDeviceEventCallBack(GtiContext * context, GtiDeviceNotificationCallBack fn) {
//    ava_argument(fn) {
//        ava_callback(_GtiDeviceNotificationCallBack);
//    }
//}

GtiDevice *GtiGetDevice(GtiContext *context, const char *devicePlatformName) {
  ava_argument(devicePlatformName) {
    ava_buffer(strlen(devicePlatformName) + 1);
    ava_in;
  }
}

GtiDevice *GtiGetAvailableDevice(GtiContext *context, GTI_DEVICE_TYPE deviceType);

// GTISDK_API int GtiLockDevice(GtiDevice *device);  // not implemented

int GtiDeviceRead(GtiDevice *device, unsigned char *buffer, unsigned int length) {
  ava_argument(buffer) {
    ava_buffer(length);
    ava_out;
  }
}

int GtiDeviceWrite(GtiDevice *device, unsigned char *buffer, unsigned int length) {
  ava_argument(buffer) {
    ava_buffer(length);
    ava_in;
  }
}

int GtiUnlockDevice(GtiDevice *device);

int GtiResetDevice(GtiDevice *device);

GTI_DEVICE_TYPE GtiGetDeviceType(GtiDevice *device);

const char *GtiGetDeviceName(GtiDevice *device) {
  const char *ret;
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
  }
}

const char *GtiGetDevicePlatformName(GtiDevice *device) {
  const char *ret;
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
  }
}

GTI_DEVICE_STATUS GtiCheckDeviceStatus(GtiDevice *device);

int GtiLoadModel(GtiDevice *device, const char *modelUrl, GTI_CHIP_MODE mode, int networkId) {
  ava_argument(modelUrl) {
    ava_buffer(strlen(modelUrl) + 1);
    ava_in;
  }
}

int GtiChangeModelMode(GtiDevice *device, GTI_CHIP_MODE mode);

int GtiChangeModelNetworkId(GtiDevice *device, int networkId);

int GtiHandleOneFrame(GtiDevice *device, unsigned char *inputBuffer, unsigned int inputLen, unsigned char *outputBuffer,
                      unsigned int outLen) {
  ava_argument(inputBuffer) {
    ava_buffer(inputLen);
    ava_in;
  }
  ava_argument(outputBuffer) {
    ava_buffer(outLen);
    ava_out;
  }
}

int GtiHandleOneFrameFloat(GtiDevice *device, unsigned char *inputBuffer, unsigned int inputLen, float *outputBuffer,
                           unsigned int outLen) {
  ava_argument(inputBuffer) {
    ava_buffer(inputLen);
    ava_in;
  }
  ava_argument(outputBuffer) {
    ava_buffer(outLen);
    ava_out;
  }
}

unsigned int GtiGetOutputLength(GtiDevice *device);

GtiDevice *GtiDeviceCreate(int DeviceType, char *FilterFileName, char *ConfigFileName) {
  ava_argument(FilterFileName) {
    ava_buffer(strlen(FilterFileName) + 1);
    ava_in;
  }

  ava_argument(ConfigFileName) {
    ava_buffer(strlen(ConfigFileName) + 1);
    ava_in;
  }
}

void GtiDeviceRelease(GtiDevice *Device);

int GtiOpenDevice(GtiDevice *Device, char *DeviceName) {
  ava_argument(DeviceName) {
    ava_buffer(strlen(DeviceName) + 1);
    ava_in;
  }
}

void GtiCloseDevice(GtiDevice *Device);

void GtiSelectNetwork(GtiDevice *Device, int NetworkId);

int GtiInitialization(GtiDevice *Device);

int GtiSendImage(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int BufferLen) {
  ava_argument(Image224Buffer) {
    ava_buffer(BufferLen);
    ava_in;
  }
}

int GtiSendImageFloat(GtiDevice *Device, float *Image224Buffer, unsigned int BufferLen) {
  ava_argument(Image224Buffer) {
    ava_buffer(BufferLen * sizeof(float));
    ava_in;
  }
}

int GtiSendTiledImage(GtiDevice *Device, unsigned char *Image224Buffer, unsigned int BufferLen) {
  ava_argument(Image224Buffer) {
    ava_buffer(BufferLen);
    ava_in;
  }
}

int GtiGetOutputData(GtiDevice *Device, unsigned char *OutputBuffer, unsigned int BufferLen) {
  ava_argument(OutputBuffer) {
    ava_buffer(BufferLen);
    ava_out;
  }
}

int GtiGetOutputDataFloat(GtiDevice *Device, float *OutputBuffer, unsigned int BufferLen) {
  ava_argument(OutputBuffer) {
    ava_buffer(BufferLen * sizeof(float));
    ava_out;
  }
}
