#include <boost/python.hpp>

#include "datset.hpp"
#include "search.hpp"
#include "discrete_projection.hpp"
#include "numeric_projection.hpp"
#include "feature_map.hpp"

//namespace np = boost::python::numpy;

/*
 * Wrapper for C++ base abstract class
 */
class projection_wrap : public projection, public p::wrapper<projection> {
public:
  const int get_total() {
    return this->get_override("get_total")();
  }
  void pprojection() {
    this->get_override("pprojection")();
  }
  void pprojection_for_file(FILE *fp) {
    this->get_override("pprojection_for_file")();
  }
  void print_header_row(FILE *fp) {
    this->get_override("print_header_row")();
  }
  double apply_projection_on_ds(Datset &ds, std::vector<int> &valid_rows, int *support) {
    return this->get_override("apply_projection_on_ds")();
  }
  bool is_projection_good_on_validation_set(Datset &ds, std::vector<int> &val_rows, int mode, double mean, double purity, int *support) {
    return this->get_override("is_projection_good_on_validation_set")();
  }
  bool do_projections_overlap(projection_wrap *pr2) {
    return this->get_override("do_projections_overlap")();
  }
  bool is_projection_better(projection *qr) {
    return this->get_override("is_projection_better")();
  }
  void copy_projection(projection_wrap *pr) {
    this->get_override("copy_projection")();
  }
  double get_projection_metric() {
	this->get_override("get_projection_metric")();
  }
};

/*
 * Python class and method declarations
 */
BOOST_PYTHON_MODULE(libfind_projections) {
  using namespace boost::python;

  Py_Initialize();
  //np::initialize();

  class_<Datset>("Datset", init<PyObject *>())
  .def("fill_datset_output_for_classification", &Datset::fill_datset_output_for_classification)
  .def("fill_datset_output_for_regression", &Datset::fill_datset_output_for_regression)
  .def("is_valid", &Datset::is_valid)
  .def("get_size", &Datset::get_rows)
  ;

  class_<search>("search")
    .def("search_projections", &search::search_projections, return_value_policy<manage_new_object>())
    .def("find_easy_explain_data", &search::find_easy_explain_data, return_value_policy<manage_new_object>())
    ;

  class_<projection_wrap, boost::noncopyable>("projection")
    .def("get_total", pure_virtual(&projection::get_total))
    .def("pprojection", &projection::pprojection)
    .def("get_att1", &projection::get_att1)
    .def("get_att2", &projection::get_att2)
    .def("get_att1_start", &projection::get_att1_start)
    .def("get_att2_start", &projection::get_att2_start)
    .def("get_att1_end", &projection::get_att1_end)
    .def("get_att2_end", &projection::get_att2_end)
    .def("get_projection_metric", pure_virtual(&projection::get_projection_metric))
	.def("point_lies_in_projection", &projection::point_lies_in_projection)
	;

  class_<discrete_projection, bases<projection> >("discrete_projection")
    .def("get_pos", &discrete_projection::get_pos)
    .def("get_neg", &discrete_projection::get_neg)
    .def("get_class", &discrete_projection::get_class)
    ;

  class_<numeric_projection, bases<projection> >("numeric_projection")
    .def("get_mean", &numeric_projection::get_mean)
    .def("get_sum_sq_error", &numeric_projection::get_sum_sq_error)
    ;

  class_<feature_map>("feature_map")
    .def("get_num_projections", &feature_map::get_num_projections)
    .def("get_projection", &feature_map::get_projection, return_value_policy<reference_existing_object>())
    ;

  class_<projection_array>("projection_array")
    .def("get_num_projections", &projection_array::get_num_projections)
    .def("get_projection", &projection_array::get_projection, return_value_policy<reference_existing_object>())
    ;
}
