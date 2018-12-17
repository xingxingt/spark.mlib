package apache.spark.mlib.huarun.datatypes
/*A distributed matrix has long-typed row and column indices and double-typed values,
 *stored distributively in one or more RDDs. It is very important to choose the right format
 *to store large and distributed matrices. Converting a distributed matrix to a different format may require a global shuffle,
 *which is quite expensive. Four types of distributed matrices have been implemented so far.
 * The basic type is called RowMatrix. A RowMatrix is a row-oriented distributed matrix without meaningful row indices,
 * e.g., a collection of feature vectors. It is backed by an RDD of its rows, where each row is a local vector.
 * We assume that the number of columns is not huge for a RowMatrix
 * so that a single local vector can be reasonably communicated to the driver and can also be stored / operated on using a single node.
 * An IndexedRowMatrix is similar to a RowMatrix but with row indices, which can be used for identifying rows and executing joins.
 * A CoordinateMatrix is a distributed matrix stored in coordinate list (COO) format, backed by an RDD of its entries;
 * A BlockMatrix is a distributed matrix backed by an RDD of MatrixBlock which is a tuple of (Int, Int, Matrix).

 *
 */
object DistributedMatrix {
  def main(args: Array[String]): Unit = {

  }

}
