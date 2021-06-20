// clang-format off
ava_name("Intel(R) Movidius(TM) Neural Compute SDK");
ava_version("2.10.01");
ava_identifier(MVNC);
ava_number(1);
ava_libs(-lmvnc);
ava_export_qualifier(dllexport);
ava_soname(libmvnc.so libmvnc.so.0);
// clang-format on

ava_non_transferable_types { ava_handle; }

struct metadata {
  unsigned long int size;
};
ava_register_metadata(struct metadata);

#include <mvnc.h>

ava_type(ncStatus_t) { ava_success(NC_OK); }

ncStatus_t ncGlobalSetOption(int option, const void *data, unsigned int dataLength) {
  ava_async;
  ava_object_record;
  ava_argument(data) {
    ava_buffer(dataLength);
    ava_in;
  }
}

ncStatus_t ncGlobalGetOption(int option, void *data, unsigned int *dataLength) {
  ava_argument(data) {
    ava_buffer(*dataLength);
    ava_out;
  }
  ava_argument(dataLength) {
    ava_buffer(1);
    ava_in;
    ava_out;
  }
}

ncStatus_t ncDeviceSetOption(struct ncDeviceHandle_t *deviceHandle, int option, const void *data,
                             unsigned int dataLength) {
  ava_async;
  ava_argument(deviceHandle) { ava_object_record; }
  ava_argument(data) {
    ava_buffer(dataLength);
    ava_input;
  }
}

ncStatus_t ncDeviceGetOption(struct ncDeviceHandle_t *deviceHandle, int option, void *data, unsigned int *dataLength) {
  ava_argument(data) {
    ava_buffer(*dataLength);
    ava_output;
  }
  ava_argument(dataLength) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
}

ncStatus_t ncDeviceCreate(int index, struct ncDeviceHandle_t **deviceHandle) {
  ava_argument(deviceHandle) {
    ava_buffer(1);
    ava_output;
    ava_element {
      ava_allocates;
      ava_object_record;
    };
  }
}

ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t *deviceHandle) {
  ava_argument(deviceHandle) { ava_object_record; }
}

ncStatus_t ncDeviceClose(struct ncDeviceHandle_t *deviceHandle) {
  ava_argument(deviceHandle) { ava_object_record; }
}

ncStatus_t ncDeviceDestroy(struct ncDeviceHandle_t **deviceHandle) {
  ava_argument(deviceHandle) {
    ava_buffer(1);
    ava_element {
      ava_object_record;
      ava_deallocates;
    };
    ava_input;
    ava_output;
  }
}

ncStatus_t ncGraphCreate(const char *ava_name, struct ncGraphHandle_t **graphHandle) {
  ava_argument(ava_name) {
    ava_buffer(strlen(ava_name) + 1);
    ava_input;
  }
  ava_argument(graphHandle) {
    ava_buffer(1);
    ava_output;
    ava_element {
      ava_object_record;
      ava_allocates;
    };
  }
}

ncStatus_t ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
                           const void *graphBuffer, unsigned int graphBufferLength) {
  ava_argument(graphHandle) {
    ava_object_record;
    ava_object_depends_on(deviceHandle);
    ava_allocates_resource(memory, graphBufferLength);
  }
  ava_argument(graphBuffer) {
    ava_buffer(graphBufferLength);
    ava_input;
  }
  ava_execute();
  ava_metadata(graphHandle)->size = graphBufferLength;
}

ncStatus_t ncGraphDestroy(struct ncGraphHandle_t **graphHandle) {
  ava_argument(graphHandle) {
    ava_buffer(1);
    ava_element {
      ava_deallocates_resource(memory, ava_metadata(*graphHandle)->size);
      ava_object_record;
    };
    ava_input;
    ava_output;
  }
}

ncStatus_t ncGraphSetOption(struct ncGraphHandle_t *graphHandle, int option, const void *data,
                            unsigned int dataLength) {
  ava_async;
  ava_argument(graphHandle) { ava_object_record; }
  ava_argument(data) {
    ava_buffer(dataLength);
    ava_input;
  }
}

ncStatus_t ncGraphGetOption(struct ncGraphHandle_t *graphHandle, int option, void *data, unsigned int *dataLength) {
  ava_argument(data) {
    ava_buffer(*dataLength);
    ava_output;
  }
  ava_argument(dataLength) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
}

ncStatus_t ncGraphQueueInference(struct ncGraphHandle_t *graphHandle, struct ncFifoHandle_t **fifoIn,
                                 unsigned int inFifoCount, struct ncFifoHandle_t **fifoOut, unsigned int outFifoCount) {
  ava_argument(fifoIn) {
    ava_buffer(inFifoCount);
    ava_input;
    ava_output;
  }
  ava_argument(fifoOut) {
    ava_buffer(outFifoCount);
    ava_input;
    ava_output;
  }
}

ncStatus_t ncGraphQueueInferenceWithFifoElem(struct ncGraphHandle_t *graphHandle, struct ncFifoHandle_t *fifoIn,
                                             struct ncFifoHandle_t *fifoOut, const void *inputTensor,
                                             unsigned int *inputTensorLength, void *userParam) {
  ava_argument(inputTensor) {
    ava_buffer(*inputTensorLength);
    ava_input;
  }
  ava_argument(inputTensorLength) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(userParam) { ava_opaque; }
}

ncStatus_t ncGraphAllocateWithFifos(struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
                                    const void *graphBuffer, unsigned int graphBufferLength,
                                    struct ncFifoHandle_t **inFifoHandle, struct ncFifoHandle_t **outFifoHandle) {
  ava_argument(graphHandle) {
    ava_object_record;
    ava_object_depends_on(deviceHandle);
  }
  ava_argument(graphBuffer) {
    ava_buffer(graphBufferLength);
    ava_input;
  }
  ava_argument(inFifoHandle) {
    ava_buffer(1);
    ava_output;
    ava_element {
      ava_object_record;
      ava_object_depends_on(deviceHandle);
    }
  }
  ava_argument(outFifoHandle) {
    ava_buffer(1);
    ava_output;
    ava_element {
      ava_object_record;
      ava_object_depends_on(deviceHandle);
    }
  }
}

ncStatus_t ncGraphAllocateWithFifosEx(struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
                                      const void *graphBuffer, unsigned int graphBufferLength,
                                      struct ncFifoHandle_t **inFifoHandle, ncFifoType_t inFifoType, int inNumElem,
                                      ncFifoDataType_t inDataType, struct ncFifoHandle_t **outFifoHandle,
                                      ncFifoType_t outFifoType, int outNumElem, ncFifoDataType_t outDataType) {
  ava_argument(graphHandle) {
    ava_object_record;
    ava_object_depends_on(deviceHandle);
  }
  ava_argument(graphBuffer) {
    ava_buffer(graphBufferLength);
    ava_input;
  }
  ava_argument(inFifoHandle) {
    ava_buffer(1);
    ava_output;
    ava_element {
      ava_object_record;
      ava_object_depends_on(deviceHandle);
    }
  }
  ava_argument(outFifoHandle) {
    ava_buffer(1);
    ava_output;
    ava_element {
      ava_object_record;
      ava_object_depends_on(deviceHandle);
    }
  }
}

ncStatus_t ncFifoCreate(const char *ava_name, ncFifoType_t ava_type, struct ncFifoHandle_t **fifoHandle) {
  ava_argument(ava_name) {
    ava_buffer(strlen(ava_name) + 1);
    ava_input;
  }
  ava_argument(fifoHandle) {
    ava_buffer(1);
    ava_output;
  }
}

ncStatus_t ncFifoAllocate(struct ncFifoHandle_t *fifoHandle, struct ncDeviceHandle_t *device,
                          struct ncTensorDescriptor_t *tensorDesc, unsigned int numElem) {
  ava_argument(tensorDesc) {
    ava_buffer(1);
    ava_input;
  }
}

ncStatus_t ncFifoSetOption(struct ncFifoHandle_t *fifoHandle, int option, const void *data, unsigned int dataLength) {
  ava_async;
  ava_argument(data) {
    ava_buffer(dataLength);
    ava_input;
  }
}

ncStatus_t ncFifoGetOption(struct ncFifoHandle_t *fifoHandle, int option, void *data, unsigned int *dataLength) {
  ava_argument(data) {
    ava_buffer(*dataLength);
    ava_input;
    ava_output;
  }
  ava_argument(dataLength) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
}

ncStatus_t ncFifoDestroy(struct ncFifoHandle_t **fifoHandle) {
  ava_argument(fifoHandle) {
    ava_buffer(1);
    ava_input;
    ava_output;
    ava_element ava_deallocates;
  }
}

ncStatus_t ncFifoWriteElem(struct ncFifoHandle_t *fifoHandle, const void *inputTensor, unsigned int *inputTensorLength,
                           void *userParam) {
  ava_argument(inputTensor) {
    ava_buffer(*inputTensorLength);
    ava_input;
  }
  ava_argument(inputTensorLength) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(userParam) { ava_opaque; }
}

ncStatus_t ncFifoReadElem(struct ncFifoHandle_t *fifoHandle, void *outputData, unsigned int *outputDataLen,
                          void **userParam) {
  ava_argument(outputData) {
    ava_buffer(*outputDataLen);
    ava_output;
  }
  ava_argument(outputDataLen) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(userParam) {
    ava_element { ava_opaque; }
    ava_buffer(1);
    ava_output;
  }
}

ncStatus_t ncFifoRemoveElem(struct ncFifoHandle_t *fifoHandle) {
  ava_argument(fifoHandle) { ava_handle; }
}
