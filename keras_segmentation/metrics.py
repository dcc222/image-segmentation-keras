import numpy as np

EPS = 1e-12

def get_iou( gt , pr , n_classes ):
	class_wise = np.zeros(n_classes)
	for cl in range(n_classes):#cl为类别class,gt为ground truth为真实标签类别，pr为predict预测的类别
		intersection = np.sum(( gt == cl )*( pr == cl ))
		union = np.sum(np.maximum( ( gt == cl ) , ( pr == cl ) ))
		iou = float(intersection)/( union + EPS )
		class_wise[ cl ] = iou
	return class_wise#返回的为每个类别的iou的数组
