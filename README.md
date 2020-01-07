  LinearSVC 与 SVC区别  
  1、LinearSVC使用的是平方hinge loss，SVC使用的是绝对值hinge loss  
  （我们知道，绝对值hinge loss是非凸的，因而你不能用GD去优化，而平方hinge loss可以）  
  2、LinearSVC使用的是One-vs-All（也成One-vs-Rest）的优化方法，而SVC使用的是One-vs-One  
  （其实我也不明白，如果有人明白恳请指教。。。）  
  3、对于多分类问题，如果分类的数量是N，则LinearSVC适合N模型，而SVC适合N(N-1)/2模型  
 （其实我也不明白+1） 
  4、对于二分类问题一般只有其中一个合适，具体问题具体对待  
  5、LinearSVC基于liblinear，罚函数是对截矩进行惩罚；SVC基于libsvm，罚函数不是对截矩进行惩罚。  
  6、我们知道SVM解决问题时，问题是分为线性可分和线性不可分问题的，liblinear对线性可分问题做了优化，故在大量数据上收敛速度比libsvm快 
  （一句话，大规模线性可分问题上LinearSVC更快）# ECG
