## EXNO-3-DS

## DATE - 12/05/2025

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/Encoding Data (2).csv")
df
```
![image](https://github.com/user-attachments/assets/265807e4-0e9c-4c68-838b-40a610b93ef9)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
```
```
e1=OrdinalEncoder(categories=[pm])
```
```
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/3d164d7a-00d1-40f1-bc83-2b25bf5f7693)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/7c0bb467-262d-42a6-b0e1-bb0fdaaca6cf)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/6302779d-b0a3-44a9-8d63-d88e7e86bd67)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
```
```
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/aa7a4a1c-dee9-4f67-9ede-673f4e74cdc1)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/49544770-62e8-44fc-9528-9737b3f42369)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/5512a634-61fb-4242-830a-bdcca6e653e7)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/data.csv")
```
```
df
```
![image](https://github.com/user-attachments/assets/dc63488d-de4a-45de-9922-3f216f3e46eb)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/45eb9d9b-96c1-4090-894b-3a7a1beea9ae)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/ee16a935-50f4-43e1-8df4-297242e1f083)
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/0a149b23-88d1-40a6-9609-4835ede10ed7)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/5603e792-d5fd-4162-83d1-21d1162d0b50)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/fe9de336-599f-4cab-aa7c-1c75e63c498f)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/99ad3f24-2717-48c5-848b-4ec01386b6cd)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3b56d433-e49e-4cf2-ac7e-72a36ba97acd)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1ef1d81a-2d16-4abd-b743-04f047b7983f)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/014d73e6-b83f-4ac4-a9ed-ad905d9b11f3)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/fd022a60-bf01-4ff6-aea1-0679fe3a5c13)
```
df["Highly Negative Skew_yeohohnson"],parameter=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Modern Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/255d889c-953b-4c05-997c-72ff6e86480c)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f0fb502c-d13e-4694-89db-590604691f3f)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/7bfa189f-3672-4ddc-aa81-fecb1d702c9e)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/a59c4272-6428-44fb-975f-320c8fb4384d)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/03d85458-5e6b-4bc2-bcc4-5cfeb19c29ab)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/81ea40e5-d256-48ea-a064-0a7f185ca38d)

# RESULT:
      ##   Thus the process of Feature Encoding and Transformation has been successfully performed on the data and the output has been attached.

       
