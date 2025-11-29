16 Nov
- started writing classifier
- simple linear and CNN
- managed to get CNN overfitting - figure out regularization next
	- train set reaches near 100%
	- test set plateaus around 70%

17 Nov
- toyed around with how the dimensions of CNN layers stack
- added batchnorm to CNN block 
- added 0.5 dropout to linear classifier head
- now nearly reaches 80% on test set - mainly helped by dropout

![Training Plot](images/Pasted%20image%2020251117130320.png)

- Added dropout to intermediate layers, increased hidden and batch sizes and got much less overfitting - actually achieves 80%, capping out at 84.7%


![Training Plot](images/Pasted%20image%2020251117144910.png)

- continued further - added weight decay and made the model bigger and got peak 88%
![Training Plot](images/Pasted%20image%2020251117182351.png)