lib_LTLIBRARIES = libxcam_core.la

PTHREAD_LDFLAGS = -pthread

XCAM_CORE_CXXFLAGS = $(XCAM_CXXFLAGS)
XCAM_CORE_LIBS = -ldl   \
	$(NULL)

if DEBUG
endif

if ENABLE_IA_AIQ
if USE_LOCAL_AIQ
IA_IMAGING_CFLAGS =  \
	-I$(top_srcdir)/ext/ia_imaging/include \
	$(NULL)

IA_IMAGING_LIBS =            \
	-L$(top_srcdir)/ext/ia_imaging/lib \
	-lia_aiq                 \
	-lia_isp_2_2             \
	-lia_cmc_parser          \
	-lia_mkn                 \
	-lia_nvm                 \
	-lia_tools               \
	-lia_log                 \
	$(NULL)

else
IA_IMAGING_CFLAGS =  $(IA_AIQ_CFLAGS)
IA_IMAGING_LIBS =  $(IA_AIQ_LIBS)
endif

XCAM_CORE_CXXFLAGS +=             \
	$(IA_IMAGING_CFLAGS)     \
	$(NULL)

XCAM_CORE_LIBS +=                 \
	$(IA_IMAGING_LIBS)       \
	$(NULL)

endif

if USE_LOCAL_ATOMISP
XCAM_CORE_CXXFLAGS +=        \
	-I$(top_srcdir)/ext/atomisp \
	$(NULL)
endif

xcam_sources =               \
        aiq3a_utils.cpp          \
	atomisp_device.cpp       \
	analyzer_loader.cpp      \
	x3a_analyzer_loader.cpp   \
	smart_analyzer_loader.cpp \
	buffer_pool.cpp          \
	device_manager.cpp       \
	dynamic_analyzer.cpp     \
	smart_analyzer.cpp       \
	smart_analysis_handler.cpp \
        fake_poll_thread.cpp \
	handler_interface.cpp    \
	image_processor.cpp      \
	isp_controller.cpp       \
	isp_image_processor.cpp  \
	isp_config_translator.cpp \
	poll_thread.cpp          \
	swapped_buffer.cpp       \
	sensor_descriptor.cpp    \
	uvc_device.cpp           \
	v4l2_buffer_proxy.cpp    \
	v4l2_device.cpp          \
	video_buffer.cpp         \
	xcam_analyzer.cpp        \
	x3a_analyzer.cpp         \
	x3a_analyzer_manager.cpp \
	x3a_analyzer_simple.cpp  \
	x3a_image_process_center.cpp  \
	x3a_stats_pool.cpp       \
	x3a_isp_config.cpp       \
	x3a_result.cpp           \
	x3a_result_factory.cpp   \
	x3a_statistics_queue.cpp \
	scaled_buffer_pool.cpp   \
	xcam_common.cpp          \
	xcam_thread.cpp          \
	x3a_analyze_tuner.cpp \
	x3a_ciq_tuning_handler.cpp \
	x3a_ciq_tnr_tuning_handler.cpp \
	x3a_ciq_bnr_ee_tuning_handler.cpp \
	$(NULL)

if ENABLE_IA_AIQ
xcam_sources +=              \
	libtbd.c                 \
	xcam_cpf_reader.c        \
	aiq_handler.cpp          \
	x3a_analyzer_aiq.cpp     \
	hybrid_analyzer.cpp      \
	$(NULL)
endif

if HAVE_LIBCL
XCAM_CORE_CXXFLAGS +=        \
	$(LIBCL_CFLAGS)          \
	-I$(top_srcdir)/clx_kernel \
	$(NULL)

XCAM_CORE_CXXFLAGS +=        \
       $(LIBCL_CFLAGS)            \
        -I$(top_srcdir)/clx_bin \
       $(NULL)

XCAM_CORE_LIBS +=            \
	$(LIBCL_LIBS)            \
	$(NULL)

xcam_sources +=              \
	cl_context.cpp           \
	cl_device.cpp            \
	cl_kernel.cpp            \
	cl_memory.cpp            \
	cl_event.cpp             \
	cl_image_bo_buffer.cpp         \
	cl_image_handler.cpp     \
	cl_image_processor.cpp   \
	cl_3a_image_processor.cpp      \
	cl_csc_image_processor.cpp    \
	cl_3a_stats_calculator.cpp     \
	cl_3a_stats_context.cpp  \
	cl_demo_handler.cpp      \
	cl_blc_handler.cpp       \
	cl_hdr_handler.cpp       \
	cl_demosaic_handler.cpp  \
	cl_csc_handler.cpp       \
	cl_wb_handler.cpp        \
	cl_denoise_handler.cpp   \
	cl_gamma_handler.cpp	 \
	cl_snr_handler.cpp	     \
	cl_macc_handler.cpp	     \
	cl_tnr_handler.cpp	     \
	cl_ee_handler.cpp	     \
	cl_dpc_handler.cpp	     \
	priority_buffer_queue.cpp	 \
	cl_bnr_handler.cpp	     \
	cl_bayer_pipe_handler.cpp    \
	cl_bayer_basic_handler.cpp   \
	cl_yuv_pipe_handler.cpp    \
	cl_rgb_pipe_handler.cpp       \
	cl_tonemapping_handler.cpp	\
	cl_newtonemapping_handler.cpp   \
	cl_biyuv_handler.cpp	 \
	cl_image_scaler.cpp   \
	cl_retinex_handler.cpp	     \
	cl_gauss_handler.cpp	     \
	$(NULL)
endif

if HAVE_LIBDRM
XCAM_CORE_CXXFLAGS += $(LIBDRM_CFLAGS)
XCAM_CORE_LIBS +=            \
	-ldrm_intel              \
	$(LIBDRM_LIBS)           \
	$(NULL)

xcam_sources +=              \
	drm_bo_buffer.cpp        \
	drm_display.cpp	         \
	drm_v4l2_buffer.cpp	 \
	$(NULL)
endif

libxcam_core_la_CXXFLAGS  =  \
	$(XCAM_CORE_CXXFLAGS)    \
	$(NULL)

libxcam_core_la_SOURCES =    \
	$(xcam_sources)          \
	$(NULL)

libxcam_core_la_LDFLAGS =    \
	-no-undefined            \
	$(PTHREAD_LDFLAGS)       \
	$(NULL)

libxcam_core_la_LIBADD =     \
	$(XCAM_CORE_LIBS)             \
	$(NULL)


libxcam_coreincludedir =  $(includedir)/xcam

nobase_libxcam_coreinclude_HEADERS =  \
	base/xcam_3a_result.h      \
	base/xcam_3a_types.h       \
	base/xcam_3a_description.h \
	base/xcam_params.h         \
	base/xcam_common.h         \
	base/xcam_defs.h           \
	base/xcam_smart_description.h \
	device_manager.h           \
	handler_interface.h        \
	image_processor.h          \
	safe_list.h                \
	smartptr.h                 \
	swapped_buffer.h           \
	v4l2_buffer_proxy.h        \
	v4l2_device.h              \
	video_buffer.h             \
	xcam_analyzer.h            \
	x3a_analyzer.h             \
	x3a_analyzer_manager.h     \
	x3a_event.h                \
	x3a_image_process_center.h \
	x3a_isp_config.h           \
	x3a_result.h               \
	xcam_mutex.h               \
	xcam_thread.h              \
	xcam_utils.h               \
	$(NULL)

if HAVE_LIBDRM
nobase_libxcam_coreinclude_HEADERS +=  \
	drm_display.h              \
	drm_v4l2_buffer.h	   \
	$(NULL)
endif
