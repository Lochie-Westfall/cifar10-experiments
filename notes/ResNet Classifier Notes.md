- started with basic architecture 
	- simple conv2d->relu blocks with slips and a linear classifier at the end
		- 64 channels
		- using adamw with no decay and lr=3e-4
		- goal today is to get it to overfit
	- No regularization - achieves 82% test and 85% train at 85 epochs in. not yet overfitting much? Maybe need to make model bigger or increase lr with scheduling or something? ![Training Plot](images/ResNet%20Exp1.png)

	- next experiment doubled res layers but halved their dimensionality 
		- this model successfully overfit to 97% train 82% test by epoch 280
		- time to move to regularization ![Training Plot](images/Pasted%20image%2020251119143005.png)
- Added batch norm but no weight decay - overfits quite badly by epoch 100 (80/90) ![Training Plot](images/Pasted%20image%2020251122092548.png)
- added 2nd convolution to resblock, added weight decay, added stride for downsampling to reduce size of final classifier and added lr scheduling
	- peaked at 86% test, 96% train after 200 epochs - still getting moderate overfitting ![Training Plot](images/Pasted%20image%2020251122125152.png)
	- added depth to model
		- now gets 89% test 99% train  ![Training Plot](images/Pasted%20image%2020251122132602.png)
- reset rate decay to 1e-4, reduced batch size from 256 to 128, reduced epochs to 100 making the scheduler work much better
	- now peaked at 92.49% test 99.9% train - severe overfitting
![Training Plot](images/Pasted%20image%2020251122140327.png)
- Added mixup and randomerasing augmentations, switch scheduler to reducelronplateu
- achieves 96% test 98% train which is close to SOTA for resnets ![Training Plot](images/Pasted%20image%2020251122165714.png)



