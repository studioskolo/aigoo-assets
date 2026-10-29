import numpy as np, cv2, sys
from PIL import Image
src,dst=sys.argv[1],sys.argv[2]
O=np.asarray(Image.open(src).convert('RGB').resize((1080,1350),Image.LANCZOS)).astype(np.float32);H,W=O.shape[:2]
# white balance: neutralize bg tint, set bg ~236 only on neutral bright pixels
bgc=O[60:400,60:400].reshape(-1,3).mean(0)
lum=O.mean(2,keepdims=True)/255; sat=(O.max(2)-O.min(2))[...,None]
w=np.clip((lum-0.55)/0.3,0,1); w=w*w*(3-2*w); w=w*np.clip((22-sat)/10,0,1)
w=cv2.GaussianBlur(w[...,0],(0,0),1.0)[...,None]
O=O+((236-bgc.mean())+(bgc.mean()-bgc))*w
O=cv2.GaussianBlur(O,(0,0),0.9)
rng=np.random.default_rng(8)
g2=rng.normal(0,1,(H,W)).astype(np.float32); g2=cv2.GaussianBlur(g2,(0,0),0.8); g2/=g2.std()
lum=O.mean(2,keepdims=True)/255; amp=3.0*(0.6+0.7*(1-np.abs(lum-0.5)*1.6).clip(0.2,1))
O=O+g2[...,None]*amp
O=cv2.GaussianBlur(O,(0,0),0.5)+rng.uniform(-0.5,0.5,O.shape)
Image.fromarray(O.clip(0,255).round().astype(np.uint8)).save(dst,quality=96,subsampling=0)
print(bgc)
