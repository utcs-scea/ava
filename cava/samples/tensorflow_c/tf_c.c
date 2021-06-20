// clang-format off
ava_name("Tensorflow C API");
ava_version("2.3.1");
ava_identifier(TF_C);
ava_number(4);
ava_libs(-ltensorflow);
ava_export_qualifier(TF_CAPI_EXPORT);
ava_soname(libtensorflow.so);
// clang-format on

ava_non_transferable_types { ava_handle; }

ava_functions { ava_time_me; }

#include <tensorflow/c/c_api.h>

ava_type(TF_Buffer) {
  TF_Buffer *ava_self;
  ava_field(data) {
    ava_buffer(ava_self->length);
    ava_lifetime_coupled(ava_self);
  }
  ava_field(length);
  ava_field(data_deallocator) /* ava_callback(buffer_deallocator, sync, ava_self) */;
}

ava_type(TF_Buffer *) { ava_buffer(1); }

ava_type(TF_Output) {
  ava_field(oper) { ava_handle; }
  ava_field(index);
}

// Status
TF_Status *TF_NewStatus(void) {
  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

void TF_DeleteStatus(TF_Status *s) {
  ava_argument(s) {
    ava_deallocates;
    ava_handle;
  }
}

void TF_SetStatus(TF_Status *s, TF_Code code, const char *msg) {
  ava_argument(msg) {
    ava_in;
    ava_buffer(strlen(msg));
  }
}

TF_Code TF_GetCode(const TF_Status *s);

const char *TF_Message(const TF_Status *s) {
  const char *ret = ava_execute();
  ava_return_value {
    ava_out;
    ava_buffer(strlen(ret) + 1);
    ava_lifetime_coupled(s);
  }
}

// Buffer
TF_Buffer *TF_NewBufferFromString(const void *proto, size_t proto_len) {
  ava_argument(proto) {
    ava_in;
    ava_buffer(proto_len);
  }
  ava_return_value {
    ava_allocates;
    ava_lifetime_manual;
    ava_out;
    ava_buffer(1);
  }
}

//#error Callers are allowed to write directly into the returned TF_Buffer
TF_Buffer *TF_NewBuffer(void) {
  ava_return_value {
    ava_allocates;
    ava_lifetime_manual;
    ava_out;
    ava_buffer(1);
  }
}

void TF_DeleteBuffer(TF_Buffer *buf) {
#warning need to find the matched buffer
  ava_unsupported;
  ava_argument(buf) {
    ava_in;
    ava_deallocates;
  }
}

/*
#warning TF_Buffer may not returned correctly
TF_Buffer
TF_GetBuffer(TF_Buffer* buffer);
*/

ava_utility void default_deallocator(void *ptr, size_t len, void *arg) {
  //    free((void *)ptr);
}

ava_callback_decl void TF_deallocator(void *data, size_t len, void *arg) {
  ava_argument(data) {
    ava_opaque;
#warning TODO: The buffer is returned opaque. This is almost certainly wrong.
    // ava_buffer(len);
  }
  ava_argument(arg) { ava_userdata; }
}

// Tensor
TF_Tensor *TF_NewTensor(TF_DataType dt, const int64_t *dims, int num_dims, void *data, size_t len,
                        void (*deallocator)(void *data, size_t len, void *arg), void *deallocator_arg) {
  ava_argument(dims) {
    ava_in;
    ava_buffer(num_dims);
  }
  ava_argument(data) {
    ava_in;
    ava_buffer(len);
  }
  ava_argument(deallocator) { ava_callback(TF_deallocator); }
  ava_argument(deallocator_arg) { ava_userdata; }
  ava_return_value {
    ava_allocates;
    ava_handle;
  }

#warning deallocator is currently overriden
  deallocator = default_deallocator;
  ava_execute();
}

/*
TF_Tensor*
TF_AllocateTensor(TF_DataType dt,
                  const int64_t* dims,
                  int num_dims,
                  size_t len)
{
    ava_argument(dims) {
        ava_in; ava_buffer(num_dims);
    }
    ava_return_value { ava_allocates; ava_handle; }
}
*/

void TF_DeleteTensor(TF_Tensor *tensor) { ava_argument(tensor) ava_deallocates; }

TF_DataType TF_TensorType(const TF_Tensor *tensor);

int TF_NumDims(const TF_Tensor *tensor);

int64_t TF_Dim(const TF_Tensor *tensor, int dim_index);

size_t TF_TensorByteSize(const TF_Tensor *);

void *TF_TensorData(const TF_Tensor *tensor) {
  ava_return_value { ava_lifetime_coupled(tensor); };
}

int64_t TF_TensorElementCount(const TF_Tensor *tensor) { ava_unsupported; }

// String
size_t TF_StringDecode(const char *src, size_t src_len, const char **dst, size_t *dst_len, TF_Status *status) {
  ava_argument(src) {
    ava_in;
    ava_buffer(src_len);
  }
  ava_argument(dst) {
    ava_out;
    ava_element ava_buffer(*dst_len);
  }
  ava_argument(status) { ava_handle; }
}

size_t TF_StringEncodedSize(size_t len);

// Session options
TF_SessionOptions *TF_NewSessionOptions(void) {
  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

void TF_SetTarget(TF_SessionOptions *options, const char *target) {
  ava_argument(target) {
    ava_in;
    ava_buffer(strlen(target));
  }
}

void TF_SetConfig(TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status) {
  ava_argument(proto) {
    ava_in;
    ava_buffer(proto_len);
  }
  ava_argument(status) { ava_handle; }
}

void TF_DeleteSessionOptions(TF_SessionOptions *options) { ava_argument(options) ava_deallocates; }

// Graph
TF_Graph *TF_NewGraph(void) {
  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

void TF_DeleteGraph(TF_Graph *graph) { ava_argument(graph) ava_deallocates; }

void TF_GraphSetTensorShape(TF_Graph *graph, TF_Output output, const int64_t *dims, const int num_dims,
                            TF_Status *status) {
  ava_argument(dims) {
    ava_in;
    ava_buffer(num_dims);
  }
  ava_argument(status) { ava_handle; }
}

TF_OperationDescription *TF_NewOperation(TF_Graph *graph, const char *op_type, const char *oper_name) {
  ava_argument(op_type) {
    ava_in;
    ava_buffer(strlen(op_type));
  }
  ava_argument(oper_name) {
    ava_in;
    ava_buffer(strlen(oper_name));
  }
  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

void TF_SetDevice(TF_OperationDescription *desc, const char *device) {
  ava_argument(device) {
    ava_in;
    ava_buffer(strlen(device));
  }
}

TF_Operation *TF_FinishOperation(TF_OperationDescription *desc, TF_Status *status) {
  ava_argument(status) ava_handle;
  ava_return_value ava_handle;
}

/*
#warning returned string may be wrong
const char*
TF_OperationName(TF_Operation* oper);

#warning returned string may be wrong
const char*
TF_OperationOpType(TF_Operation* oper);

#warning returned string may be wrong
const char*
TF_OperationDevice(TF_Operation* oper);
*/

int TF_OperationNumOutputs(TF_Operation *oper);

int TF_OperationNumInputs(TF_Operation *oper);

TF_Operation *TF_GraphOperationByName(TF_Graph *graph, const char *oper_name) {
  ava_argument(oper_name) {
    ava_in;
    ava_buffer(strlen(oper_name) + 1);
  }
  ava_return_value ava_handle;
}

TF_Operation *TF_GraphNextOperation(TF_Graph *graph, size_t *pos) {
  ava_argument(pos) {
    ava_out;
    ava_buffer(1);
  }
  ava_return_value ava_handle;
}

TF_ImportGraphDefOptions *TF_NewImportGraphDefOptions(void) {
  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions *opts) { ava_argument(opts) ava_deallocates; }

void TF_GraphImportGraphDef(TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options,
                            TF_Status *status) {
  ava_argument(graph_def) {
    ava_in;
    ava_buffer(1);
    /* ava_element ava_field(data) ava_buffer(graph_def->length); */
  }
  ava_argument(status) { ava_handle; }
}

// Session
TF_Session *TF_NewSession(TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status) {
  ava_argument(status) { ava_handle; }
  ava_return_value {
    ava_allocates;
    ava_handle;
  }
}

void TF_CloseSession(TF_Session *sess, TF_Status *status) {
  ava_argument(status) { ava_handle; }
}

void TF_DeleteSession(TF_Session *sess, TF_Status *status) {
  ava_argument(sess) ava_deallocates;
  ava_argument(status) { ava_handle; }
}

void TF_SessionRun(TF_Session *session,
                   // RunOptions
                   const TF_Buffer *run_options,
                   // Input tensors
                   const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs,
                   // Output tensors
                   const TF_Output *outputs, TF_Tensor **output_values, int noutputs,
                   // Target operations
                   const TF_Operation *const *target_opers, int ntargets,
                   // RunMetadata
                   TF_Buffer *run_metadata,
                   // Output
                   TF_Status *status) {
  ava_argument(run_options) {
    ava_in;
    ava_buffer(1);
  }
  ava_argument(inputs) {
    ava_in;
    ava_buffer(ninputs);
  }
  ava_argument(input_values) {
    ava_in;
    ava_buffer(ninputs);
  }
  ava_argument(outputs) {
    ava_in;
    ava_buffer(noutputs);
  }
  ava_argument(output_values) {
    ava_out;
    ava_buffer(noutputs);
  }
  ava_argument(target_opers) {
    ava_in;
    ava_buffer(ntargets);
    ava_element ava_handle;
  }
  ava_argument(run_metadata) {
    ava_out;
    ava_buffer(1);
  }
  ava_argument(status) { ava_handle; }
}

const char *TF_Version() {
  const char *ret = ava_execute();
  ava_return_value {
    ava_out;
    ava_lifetime_static;
    ava_buffer(strlen(ret) + 1);
  }
}

size_t TF_StringEncode(const char *src, size_t src_len, char *dst, size_t dst_len, TF_Status *status) {
  ava_argument(src) {
    ava_buffer(src_len);
    ava_input;
  }
  ava_argument(dst) {
    ava_buffer(dst_len);
    ava_output;
  }
  ava_argument(status) { ava_handle; }
}
