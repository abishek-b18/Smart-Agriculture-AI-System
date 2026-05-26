import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import pickle


data = pd.read_excel(
"Smart_Agriculture_Datasets.xlsx",
sheet_name="crop_recommendation"
)


X = data.drop(
'label',
axis=1
)


y = data['label']


model = RandomForestClassifier()

model.fit(
X,
y
)


pickle.dump(

model,

open(
'models/crop_model.pkl',
'w b'
)

)


print(

"Model Saved"

)