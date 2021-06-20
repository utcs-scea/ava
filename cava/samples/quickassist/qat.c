ava_name("Intel® QuickAssist Technology");
ava_version("1.7");
ava_identifier(QAT);
ava_number(6);
ava_cflags(-I$ENV{ICP_ROOT} / quickassist / include -
           I$ENV{ICP_ROOT} / quickassist / lookaside / access_layer / include);
ava_libs(-lqat_s - lusdm_drv_s);
ava_export_qualifier();
ava_soname(libqat_s.so);

ava_begin_utility;
//#include <qae_mem.h>
#include <stdint.h>

void *qaeMemAllocNUMA(uint32_t size, uint32_t node, uint32_t alignment);
void qaeMemFreeNUMA(void **ptr);
uint64_t qaeVirtToPhysNUMA(void *);
ava_end_utility;

ava_begin_replacement;

uint64_t qaeVirtToPhysNUMA(void *p) { return (uint64_t)ava_zerocopy_get_physical_address(p); }
void *qaeMemAllocNUMA(uint32_t size, uint32_t node, uint32_t alignment) { return ava_zerocopy_alloc(size); }
void qaeMemFreeNUMA(void **ptr) {
  ava_zerocopy_free(*ptr);
  *ptr = NULL;
}
// TODO: Make sure zero-copy buffers are transferred correctly to the worker.
ava_end_replacement;

#include <cpa.h>
#include <dc/cpa_dc.h>
#include <dc/cpa_dc_bp.h>
#include <dc/cpa_dc_dp.h>
#include <icp_sal_poll.h>
#include <icp_sal_user.h>
//#include <icp_adf_accel_mgr.h>

ava_begin_utility;
CpaStatus icp_adf_get_numDevices(Cpa32U *num_dev);
ava_end_utility;

typedef struct {
  uint32_t pPrivateMetaDataSize;
  Cpa32U pSessionSize;
  void (*instance_callbackFn)(void *, CpaStatus);
} QATMetadata;

ava_register_metadata(QATMetadata);

ava_utility QATMetadata global_metadata;

/* Throughput resource, represent for compressionDataRate */
ava_throughput_resource qat_throughput;

ava_utility void *QATAVAMalloc(size_t size) {
  void *ret = qaeMemAllocNUMA(size, 0, 64);
  assert(ret);
  return ret;
}

ava_utility void QATAVAFree(void *ptr) { qaeMemFreeNUMA(&ptr); }

/*
ava_functions {
    ava_time_me;
}
*/

ava_type(CpaFlatBuffer) {
  CpaFlatBuffer *ava_self;
  ava_field(pData) {
    ava_opaque;
    //        //ava_buffer(ava_self->dataLenInBytes);
    //        ava_buffer(74752);
    //        ava_buffer_allocator(QATAVAMalloc, QATAVAFree);
    //        // Mark these as never being deallocated. Ideally these would be
    //        // coupled (transitively) to a CpaBufferList which could be freed.
    //        ava_lifetime_static;
    //        // AMP: If ava_{in,out}put were value annotations they should be placed here in a sense. However they
    //        would actually need to be added at the call. Which is pain, but #define will help.
  }
}

ava_type(CpaBufferList) {
  CpaBufferList *ava_self;
  ava_field(pBuffers) {
    ava_buffer(ava_self->numBuffers);
    ava_lifetime_static;
  }
  ava_field(pUserData) { ava_opaque; }
  ava_field(pPrivateMetaData) {
    ava_opaque;
    //    ava_buffer(global_metadata.pPrivateMetaDataSize);
    //    ava_buffer_allocator(QATAVAMalloc, QATAVAFree);
    //    ava_lifetime_static;
  }
}

CpaStatus cpaDcQueryCapabilities(CpaInstanceHandle dcInstance, CpaDcInstanceCapabilities *pInstanceCapabilities) {
  ava_argument(dcInstance) {
    ava_handle;
    ava_input;
  }
  ava_argument(pInstanceCapabilities) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
}

ava_callback_decl void cpa_callbackFn(void *userdata, CpaStatus status) {
  ava_argument(userdata) { ava_userdata; }
}

CpaStatus cpaDcInitSession(CpaInstanceHandle dcInstance, CpaDcSessionHandle pSessionHandle,
                           CpaDcSessionSetupData *pSessionData, CpaBufferList *pContextBuffer,
                           void (*callbackFn)(void *, CpaStatus)) {
  ava_sync;

  ava_argument(dcInstance) {
    ava_handle;
    ava_input;
  }
  ava_argument(pSessionHandle) {
    ava_buffer(global_metadata.pSessionSize);
    ava_buffer_allocator(QATAVAMalloc, QATAVAFree);
    // Tell AvA that the lifetime of this buffer is coupled to dcInstance instead of being only the call.
    ava_lifetime_manual;
    // There is no need to copy this buffer either way.
    ava_no_copy;
  }
  ava_argument(pSessionData) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(pContextBuffer) {
    ava_buffer(1);
    ava_input;
  }
  ava_argument(callbackFn) { ava_callback_registration(cpa_callbackFn); }

  ava_metadata(dcInstance)->instance_callbackFn = callbackFn;
}

CpaStatus cpaDcRemoveSession(const CpaInstanceHandle dcInstance, CpaDcSessionHandle pSessionHandle) {
  ava_argument(dcInstance) {
    ava_handle;
    ava_input;
  }
  ava_argument(pSessionHandle) {
    ava_buffer(global_metadata.pSessionSize);
    ava_buffer_allocator(QATAVAMalloc, QATAVAFree);
    ava_lifetime_manual;
    ava_deallocates;
    ava_no_copy;
  }
}

ava_utility uint32_t calc_CpaBufferList_size(CpaBufferList *bufferList) {
  uint32_t data_size = 0;
  for (int i = 0; i < bufferList->numBuffers; i++) {
    data_size += bufferList->pBuffers[i].dataLenInBytes;
  }
  return data_size;
}

CpaStatus cpaDcCompressData2(CpaInstanceHandle dcInstance, CpaDcSessionHandle pSessionHandle, CpaBufferList *pSrcBuff,
                             CpaBufferList *pDestBuff, CpaDcOpData *pOpData, CpaDcRqResults *pResults,
                             void *callbackTag) {
  // ava_time_me;
  ava_implicit_argument void (*callback)(void *, int32_t) = ava_metadata(dcInstance)->instance_callbackFn;
  ava_argument(callback) { ava_callback(cpa_callbackFn); }

  // AMP: Same comments as cpaDcDecompressData

  ava_argument(dcInstance) {
    ava_handle;
    ava_input;
  }
  ava_argument(pSessionHandle) {
    ava_buffer(global_metadata.pSessionSize);
    ava_buffer_allocator(QATAVAMalloc, QATAVAFree);
    ava_lifetime_manual;
    ava_no_copy;
  }
  ava_argument(pSrcBuff) {
    ava_buffer(1);
    ava_input;
  }
  ava_argument(pDestBuff) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(pOpData) {
    ava_buffer(1);
    ava_input;
  }
  ava_argument(pResults) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(callbackTag) { ava_userdata; }
  ava_execute();
  ava_consumes_resource(qat_throughput, calc_CpaBufferList_size(pSrcBuff));
}

CpaStatus cpaDcDecompressData(CpaInstanceHandle dcInstance, CpaDcSessionHandle pSessionHandle, CpaBufferList *pSrcBuff,
                              CpaBufferList *pDestBuff, CpaDcRqResults *pResults, CpaDcFlush flushFlag,
                              void *callbackTag) {
  ava_implicit_argument void (*callback)(void *, int32_t) = ava_metadata(dcInstance)->instance_callbackFn;
  ava_argument(callback) { ava_callback(cpa_callbackFn); }

  // AMP: This should be marked as ava_async to get the best performance in async mode.

  ava_argument(dcInstance) {
    ava_handle;
    ava_input;
  }
  ava_argument(pSessionHandle) {
    ava_buffer(global_metadata.pSessionSize);
    ava_buffer_allocator(QATAVAMalloc, QATAVAFree);
    ava_lifetime_manual;
    ava_no_copy;
  }
  ava_argument(pSrcBuff) {
    ava_buffer(1);
    ava_input;
  }
  ava_argument(pDestBuff) {
    ava_buffer(1);
    ava_input;   // AMP: This being ava_input makes all dest buffers be copied in as well as out.
    ava_output;  // AMP: This should not be an out buffer in async mode.
  }
  ava_argument(pResults) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  ava_argument(callbackTag) { ava_userdata; }

  // AMP: In async mode, pDestBuff needs to be stored in ava_metadata(callbackTag) and transported as an implicit
  // argument of the callback. Otherwise the result data will never be copied back to the guest.
}

CpaStatus cpaDcGetNumInstances(Cpa16U *pNumInstances) {
  ava_sync;

  ava_argument(pNumInstances) {
    ava_buffer(1);
    ava_output;
  }
}

CpaStatus cpaDcGetInstances(Cpa16U numInstances, CpaInstanceHandle *dcInstances) {
  ava_sync;
  ava_argument(dcInstances) {
    ava_output;
    ava_buffer(numInstances);
    ava_element ava_handle;
  }
}

CpaStatus cpaDcGetNumIntermediateBuffers(CpaInstanceHandle instanceHandle, Cpa16U *pNumBuffers) {
  ava_sync;

  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
    ava_output;
  }
  ava_argument(pNumBuffers) {
    ava_buffer(1);
    ava_output;
  }
}

CpaStatus cpaDcStartInstance(CpaInstanceHandle instanceHandle, Cpa16U numBuffers,
                             CpaBufferList **pIntermediateBuffers) {
  ava_sync;

  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
    ava_output;
  }
  ava_argument(pIntermediateBuffers) {
    ava_buffer(numBuffers);
    ava_element ava_buffer(1);
    ava_lifetime_coupled(instanceHandle);
    ava_no_copy;
    ava_input;
  }
}

CpaStatus cpaDcStopInstance(CpaInstanceHandle instanceHandle) {
  ava_sync;

  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
    // Mark the deallocation so that coupled things get deallocated
    ava_deallocates;
  }
}

CpaStatus cpaDcInstanceGetInfo2(const CpaInstanceHandle instanceHandle, CpaInstanceInfo2 *pInstanceInfo2) {
  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
  }
  ava_argument(pInstanceInfo2) {
    ava_buffer(1);
    ava_output;
  }
}

CpaStatus cpaDcGetSessionSize(CpaInstanceHandle dcInstance, CpaDcSessionSetupData *pSessionData, Cpa32U *pSessionSize,
                              Cpa32U *pContextSize) {
  ava_argument(dcInstance) {
    ava_handle;
    ava_input;
  }
  ava_argument(pSessionData) {
    ava_buffer(1);
    ava_input;
  }
  ava_argument(pSessionSize) {
    ava_buffer(1);
    ava_output;
  }
  ava_argument(pContextSize) {
    ava_buffer(1);
    ava_output;
  }
  CpaStatus ret = ava_execute();
  global_metadata.pSessionSize = *pSessionSize;
}

CpaStatus cpaDcBufferListGetMetaSize(const CpaInstanceHandle instanceHandle, Cpa32U numBuffers, Cpa32U *pSizeInBytes) {
  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
  }
  ava_argument(pSizeInBytes) {
    ava_buffer(1);
    ava_input;
    ava_output;
  }
  CpaStatus ret = ava_execute();
  global_metadata.pPrivateMetaDataSize = *pSizeInBytes;
}

CpaStatus cpaDcSetAddressTranslation(const CpaInstanceHandle instanceHandle,
                                     CpaPhysicalAddr (*virtual2Physical)(void *)) {
  ava_sync;

  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
  }
  virtual2Physical = &qaeVirtToPhysNUMA;
  ava_execute();
}

CpaStatus icp_sal_DcPollInstance(CpaInstanceHandle instanceHandle, Cpa32U response_quota) {
  ava_argument(instanceHandle) {
    ava_handle;
    ava_input;
  }
}

CpaStatus icp_sal_userStartMultiProcess(const char *pProcessName, CpaBoolean limitDevAccess) {
  ava_argument(pProcessName) {
    ava_buffer(strlen(pProcessName) + 1);
    ava_input;
  }
}

CpaStatus icp_sal_userStop(void) {}

CpaStatus icp_adf_get_numDevices(Cpa32U *num_dev) { ava_sync; }
