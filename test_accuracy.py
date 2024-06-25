from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler




def parse_args():
    parser = argparse.ArgumentParser(description="get attack accuracy")
    parser.add_argument("--target_member_dir",type=str,default=None)
    parser.add_argument("--target_non_member_dir",type=str,default=None)
    parser.add_argument("--shadow_member_dir",type=str,default=None)
    parser.add_argument("--shadow_non_member_dir",type=str,default=None)
    parser.add_argument("--method",type=str,default="classifier")
    args = parser.parse_args()

    return args

def process_data():
    t_m = torch.load(args.target_member_dir)
    t_n_m = torch.load(args.target_non_member_dir)

    s_m = torch.load(args.shadow_member_dir)
    s_n_m = torch.load(args.shadow_non_member_dir)


    train_datasets = [s_m, s_n_m]
    test_datasets = [t_m, t_n_m]

    train_features = []
    train_labels = []

    test_features = []
    test_labels = []
    for dataset in train_datasets:
        for item in dataset:
            feature = item[0]
            label = int(item[1])
            train_features.append(feature)
            train_labels.append(label)

    for dataset in test_datasets:
        for item in dataset:
            feature = item[0]
            label = int(item[1])
            test_features.append(feature)
            test_labels.append(label)
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    return train_features,train_labels,test_features,test_labels


class DefineClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(DefineClassifier, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


        self.layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 2
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 3
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        # Layer 4
        self.layer5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU6(),
            nn.Dropout(0.5)
        )
        
        self.out_layer = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return self.out_layer(x)






def main(train_features,train_labels,test_features,test_labels):
    method = args.method
    if method=="classifier":
        train_labels = torch.tensor(train_labels) 
        train_features = torch.tensor(train_features)

        test_labels = torch.tensor(test_labels) 
        test_features = torch.tensor(test_features)
        input_dim = train_features.shape[1]
        train_features = train_features.to(torch.float32)
        test_features = test_features.to(torch.float32)
        model = DefineClassifier(input_dim)
        print(train_labels.dtype) 
        print(next(model.parameters()).dtype) 

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  

        num_epochs = 1000
        cur_auc = 0
        TPR1 = -1
        TPR_01 = -1
        best_test = 0
        for epoch in range(num_epochs):
            model.train()  
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, train_predicted = torch.max(outputs.data, 1)
            train_correct = (train_predicted == train_labels).sum().item()
            train_accuracy = train_correct / train_labels.size(0) * 100
            
            model.eval() 
            with torch.no_grad():
                test_outputs = model(test_features)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_correct = (test_predicted == test_labels).sum().item()
                test_accuracy = test_correct / test_labels.size(0) * 100
            best_test = best_test if best_test > test_accuracy else test_accuracy
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}%, Best_test: {best_test:.4f}%, AUC: {cur_auc:.4f}%, TPR1%: {TPR1:.4f},TPR0.1%{TPR_01:.4f}")
            
            if (epoch + 1) % 1 == 0: 
                fpr, tpr, thresholds = roc_curve(test_labels.numpy(), test_outputs[:, 1].numpy())
                roc_auc = auc(fpr, tpr)
                cur_auc = roc_auc if roc_auc > cur_auc else cur_auc
                desired_fpr_threshold = 0.01
                idx = np.argmin(np.abs(fpr - desired_fpr_threshold))
                tpr_at_desired_fpr = tpr[idx]
                TPR1 = tpr_at_desired_fpr if tpr_at_desired_fpr > TPR1 else TPR1
                desired_fpr_threshold = 0.001
                idx = np.argmin(np.abs(fpr - desired_fpr_threshold))
                tpr_at_desired_fpr = tpr[idx]
                TPR_01 = tpr_at_desired_fpr if tpr_at_desired_fpr > TPR_01 else TPR_01
                print(f"TPR at FPR = {desired_fpr_threshold*100}%: {tpr_at_desired_fpr:.4f}")
    elif method=="distribution":
        class0_samples = train_features[train_labels == 0]
        mean_0 = np.mean(class0_samples, axis=0)
        cov_0 = np.cov(class0_samples, rowvar=False)
        class1_samples = train_features[train_labels == 1]
        mean_1 = np.mean(class1_samples, axis=0)
        cov_1 = np.cov(class1_samples, rowvar=False)
        reg_value = 1e-5
        cov_0 += reg_value * np.eye(cov_0.shape[0])
        cov_1 += reg_value * np.eye(cov_1.shape[0])
        rv_0 = multivariate_normal(mean_0, cov_0)
        rv_1 = multivariate_normal(mean_1, cov_1)
        predictions = []
        for x in test_features:
            p_0 = rv_0.logpdf(x)
            p_1 = rv_1.logpdf(x)
            if p_0 > p_1:
                predictions.append(0)
            else:
                predictions.append(1)
        scores = [p_1 - p_0 for p_0, p_1 in zip(rv_0.logpdf(test_features), rv_1.logpdf(test_features))]

        auc_roc = roc_auc_score(test_labels, scores)
        print(f"AUC-ROC: {auc_roc:.2f}")

        desired_fpr = 0.01  
        fpr, tpr, _ = roc_curve(test_labels, scores)
        index = next(i for i, f in enumerate(fpr) if f > desired_fpr) - 1
        print(f"TPR at FPR {desired_fpr}: {tpr[index]:.2f}")
        accuracy = np.mean(predictions == test_labels)
        print(f"Accuracy: {accuracy:.2f}")
    elif method=="threshold":
        data_means = np.max(train_features, axis=1)

        sorted_means = np.sort(data_means)

        potential_thresholds = (sorted_means[:-1] + sorted_means[1:]) / 2
        accuracies = []
        for threshold in potential_thresholds:
            predicted_labels = np.where(data_means > threshold, 1, 0)
            accuracy = np.mean(predicted_labels == train_labels)
            accuracies.append(accuracy)

        best_threshold = potential_thresholds[np.argmax(accuracies)]
        print(f"Best Threshold: {best_threshold}")

        data_means_test = np.max(test_features, axis=1)
        labels = np.where(data_means_test > best_threshold, 1, 0)
        accuracy = np.mean(labels == test_labels)
        print(f"Accuracy with Best Threshold: {accuracy:.2f}")

        fpr, tpr, thresholds = roc_curve(test_labels, data_means_test)
        roc_auc = auc(fpr, tpr)
        print(f"AUC-ROC: {roc_auc:.3f}")

        desired_fpr = 0.01
        closest_fpr_index = np.argmin(np.abs(fpr - desired_fpr))
        tpr_at_desired_fpr = tpr[closest_fpr_index]
        print(f"TPR at FPR {desired_fpr}: {tpr_at_desired_fpr:.2f}")


if __name__ == "__main__":
    args = parse_args()
    train_features,train_labels,test_features,test_labels = process_data()
    main(train_features,train_labels,test_features,test_labels)
