from sklearn.preprocessing import StandardScaler, MaxAbsScaler, PolynomialFeatures,RobustScaler
from  sklearn.decomposition import PCA
# Preprocessings
SC = [StandardScaler(), "SC"]
MAS = [MaxAbsScaler(), "MAS"]
PolFeat = [PolynomialFeatures(degree=3), "PolFeat"]
RobSc = [RobustScaler(), "RobSc"]

# Feature selections
PCA = [PCA(), "PCA"]
