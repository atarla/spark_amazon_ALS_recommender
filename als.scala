import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

//dataset location directory
val data_dir = "/home/ec2-user/dataset/"

//get the file and verify count
val dat = sc.textFile(data_dir + "amazon-reviews-ratings.txt")
val data = dat.map(_.split("::") match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})

data.count

//To reverse reviewer and product id un-commment the below lines

//val data = sc.textFile(data_dir + "amazon_test.txt").map { line =>
//  val fields = line.split("::")
//  Rating(fields(1).toInt, fields(0).toInt, fields(2).toDouble)
//}

val ranks = 10
val lambdas = 0.01
val numIters = 10
var bestModel: Option[MatrixFactorizationModel] = None
val model = ALS.train(data, ranks, numIters, lambdas)
val usersProducts = data.map { case Rating(user, product, rate) =>
      (user, product)
    }
val recommendations = model.predict(usersProducts).map{ case Rating(user,product,rate) => ((user,product),rate)}

val ratesAndPreds = data.map { case Rating(user, product, rate) =>((user, product), rate)}.join(recommendations)
val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) => 
	val err = (r1 - r2)
	err * err}
	.mean()
println("Mean Squared Error = " + MSE)

    // Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter")
val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
println("Product recommended for you:")

//Print 10 users that are most likely to buy product with ID-158
model.recommendProducts(158,10)

//Print 10 products that the reviewer is most likely to buy
model.recommendUsers(729404,10

