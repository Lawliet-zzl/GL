"""
The codes are based on the original codes from https://github.com/ShiyuLiang/odin-pytorch/blob/master/code/calMetric.py
Created on Wed Dec 16 2020
@author: xxx
"""
from __future__ import print_function
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def Calibrated(dataloader, net, bins=20, temperature = 1, dtype = 'ECE'):
	net.eval()
	if dtype == 'ECE':
		ece_criterion = ECELoss(n_bins=bins, temperature = temperature).cuda()
	elif dtype == 'AdaECE':
		ece_criterion = AdaECELoss(n_bins=bins, temperature = temperature).cuda()

	logits_list = []
	labels_list = []
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs = net(inputs)
			logits_list.append(outputs.cpu())
			labels_list.append(targets.cpu())
		logits = torch.cat(logits_list).cuda()
		labels = torch.cat(labels_list).cuda()
	ece, ys = ece_criterion(logits, labels)
	ece = ece.item()
	return ece, ys


class ECELoss(nn.Module):
	def __init__(self, n_bins=20, temperature = 1):
		super(ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]
		self.temperature = temperature
	def forward(self, logits, labels):
		softmaxes = F.softmax(logits / self.temperature, dim=1)
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=logits.device)
		ys = []
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
				accuracy_in_bin = accuracy_in_bin.item()
			else:
				accuracy_in_bin = 0
			ys.append(accuracy_in_bin)
		return ece, ys

class AdaECELoss(nn.Module):
	def __init__(self, n_bins=20, temperature = 1):
		super(AdaECELoss, self).__init__()
		self.n_bins = n_bins
		self.temperature = temperature
	def forward(self, logits, labels):
		softmaxes = F.softmax(logits / self.temperature, dim=1)
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=logits.device)
		ys = []

		confidences, indices = torch.sort(confidences)
		accuracies = accuracies[indices]

		num = softmaxes.size(0)
		window = int(num / self.n_bins)

		avg_confidence_in_bin = torch.zeros(1, device=logits.device)
		accuracy_in_bin = torch.zeros(1, device=logits.device)
		for i in range(num):
			avg_confidence_in_bin += confidences[i]
			accuracy_in_bin += accuracies[i]
			if (i + 1) % window == 0:
				avg_confidence_in_bin = avg_confidence_in_bin / window
				accuracy_in_bin = accuracy_in_bin / window
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * window / num
				ys.append(accuracy_in_bin.item())
				avg_confidence_in_bin = torch.zeros(1, device=logits.device)
				accuracy_in_bin = torch.zeros(1, device=logits.device)
		return ece, ys

def tpr95(soft_IN, soft_OOD, precision):
	#calculate the falsepositive error when tpr is 95%

	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision # precision:200000

	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			total += 1
	if total == 0:
		fprBase = 1
	else:
		fprBase = fpr/total
	return fprBase

def auroc(soft_IN, soft_OOD, precision):
	#calculate the AUROC
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr
	#improve
	return aurocBase

def auroc_XY(soft_IN, soft_OOD, precision):
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	tprs = []
	fprs = []
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		tprs.append(tpr)
		fprs.append(fpr)
	return tprs, fprs

def auprIn(soft_IN, soft_OOD, precision):
	#calculate the AUPR

	precisionVec = []
	recallVec = []
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(start, end, gap):
		tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
		if tp + fp == 0: continue
		precision = tp / (tp + fp)
		recall = tp
		precisionVec.append(precision)
		recallVec.append(recall)
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def auprOut(soft_IN, soft_OOD, precision):
	#calculate the AUPR
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(end, start, -gap):
		fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
		if tp + fp == 0: break
		precision = tp / (tp + fp)
		recall = tp
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def detection(soft_IN, soft_OOD, precision):
	#calculate the minimum detection error
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	errorBase = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

	return errorBase

def get_softmax_MRL(net_C, net_D, eps, std, dataloader):

	net_C.eval()
	net_D.eval()

	res = np.array([])

	total = 0
	cnt = 0

	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()

			outputs_C = net_C(inputs)
			outputs_D = net_D(inputs)

			num_classes = outputs_C.size(1)
			total += outputs_C.size(0)

			A = torch.mul(outputs_C, outputs_D).sum(dim = 1, keepdim = True).repeat(1, num_classes) * outputs_C
			B = torch.matmul(outputs_D, std)
			Coupling = eps*(A + B) 

			softmax_vals, predicted = torch.max(F.softmax(Coupling.data, dim=1), dim=1)

			res = np.append(res, softmax_vals.cpu().numpy())

	return res

def get_softmax(net, dataloader):
	net.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs = net(inputs)
			softmax_vals, predicted = torch.max(F.softmax(outputs.data, dim=1), dim=1)
			res = np.append(res, softmax_vals.cpu().numpy())
	return res
		
def detect_OOD(soft_ID, soft_OOD, precision=200000):

	detection_results = np.array([0.0,0.0,0.0,0.0,0.0])
	detection_results[0] = auroc(soft_ID, soft_OOD, precision)*100
	detection_results[1] = auprIn(soft_ID, soft_OOD, precision)*100
	detection_results[2] = auprOut(soft_ID, soft_OOD, precision)*100
	detection_results[3] = tpr95(soft_ID, soft_OOD, precision)*100
	detection_results[4] = detection(soft_ID, soft_OOD, precision)*100

	return detection_results