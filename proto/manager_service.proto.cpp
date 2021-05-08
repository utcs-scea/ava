#include "manager_service.proto.h"

namespace {

zpp::serializer::register_types<
    zpp::serializer::make_type<ava_proto::WorkerAssignRequest,
                               zpp::serializer::make_id("v1::ava_proto::WorkerAssignRequest")>,
    zpp::serializer::make_type<ava_proto::WorkerAssignReply,
                               zpp::serializer::make_id("v1::ava_proto::WorkerAssignReply")> >
    _;

}  // namespace
