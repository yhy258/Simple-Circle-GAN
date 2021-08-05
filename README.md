# 설명
https://www.notion.so/CircleGAN-b94ade4e054b47269152517849973439
  
## Note
연구를 좀 해봤는데 Iteration이 지날 수록 lr scheduler의 역할이 굉장히 큰 것 같다.  
이 코드에는 Lr scheduler가 없으니, 사용 시 첨가 해야함.  
official의 경우, max_iter/10 step마다 exponential decay함.  

# Result
<img src="https://github.com/yhy258/Simple-Circle-GAN/blob/master/circlegan_cifarresult_45000_iter.png?raw=true" width="500" height="500">
