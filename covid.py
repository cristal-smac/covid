import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from deap import base
from deap import creator
from deap import tools
import array
from deap import algorithms

# Collecte et mise en forme des données

def collecte_donnees(granularite='national'):

	data_brute = pd.read_csv('chiffres-cles.csv')
	dates = pd.read_csv('dates.csv')

	# National
	if granularite == "national":
		data_brute = data_brute.loc[data_brute['source_type'] == "ministere-sante"]
		data_brute = data_brute.loc[data_brute['granularite'] == "pays"].reset_index()
		data_brute = data_brute[['date','cas_confirmes', 'deces', 'deces_ehpad', 'reanimation', 'hospitalises', 'gueris']]
		data_brute = dates.set_index('date').join(data_brute.set_index('date'))
		data = data_brute.interpolate(limit_area='inside')
		data = data.dropna(how='all')
		data['reanimation'][0:3] = 0
		data['gueris'][0:3] = 0

	# Regional
	if granularite == "regional":
		data_brute = data_brute.loc[data_brute['source_type'] == "opencovid19-fr"]
		data_brute = data_brute.loc[data_brute['maille_code'] == "REG-32"].reset_index()
		data_brute = data_brute[['date','cas_confirmes', 'deces', 'deces_ehpad', 'reanimation', 'hospitalises', 'gueris']]
		data_brute = dates.set_index('date').join(data_brute.set_index('date'))
		data = data_brute.interpolate(limit_area='inside')
		data = data.dropna(how='all')
		data = data.loc['2020-03-18':]

	# Departemental
	if granularite == "departemental":
		data_brute = pd.read_csv(glob.glob('donnees-hospitalieres-covid19-*')[-1], sep=";")
		data_brute = data_brute.loc[data_brute['dep'] == "59"]
		data_brute[data_brute['sexe']==0].groupby(['jour']).sum()
		data = data.drop(columns=['sexe'])

	return data

# Optimisation génétique

def optimisation_genetique(n_indiv, n_gen, data_objectif, periodes_confinement, population, fonction_eval, nb_parametres, verbose=False):

	creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("attr_float", random.random)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=nb_parametres)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("evaluate", fonction_eval, data_objectif=data_objectif, periodes_confinement=periodes_confinement, population=population)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
	toolbox.register("select", tools.selTournament, tournsize=3)

	pop = toolbox.population(n=n_indiv)
	hof = tools.HallOfFame(1)
	
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, 
								   stats=stats, halloffame=hof, verbose=verbose)
	return hof[0]

# Modèles à compartiments et leur fonction d'évaluation

def SIGRM_increment(S, I, G, R, M, liste_tauxTransmiss, tauxRemis, tauxGraves, tauxRemisGraves, tauxMortalite):
	total_rea = 0
	for tauxTransmiss in liste_tauxTransmiss:

		nouveauxCas = tauxTransmiss*S[-1]*I[-1]		
		nouveauxInfectesRemis=tauxRemis*I[-1]
		nouveauxInfectesGraves=tauxGraves*I[-1]
		total_rea += nouveauxInfectesGraves
		
		nouveauxGravesRemis=tauxRemisGraves*G[-1]
		
		nouveauxGravesMorts=tauxMortalite*G[-1]
		
		S.append(S[-1]-nouveauxCas) 
		I.append(I[-1]+nouveauxCas-nouveauxInfectesRemis-nouveauxInfectesGraves)
		G.append(G[-1]+nouveauxInfectesGraves-nouveauxGravesMorts-nouveauxGravesRemis)
		R.append(R[-1]+nouveauxInfectesRemis+nouveauxGravesRemis)
		M.append(M[-1]+nouveauxGravesMorts)
		assert(round(S[-1]+I[-1]+G[-1]+R[-1]+M[-1],3) == 1) # la somme fait toujours 1
	
	return total_rea

def eval_SIGRM(candidat, data_objectif, periodes_confinement, population=1, plot=False, altered=False):
	candidat = np.absolute(candidat)

	tauxInfecteInitial = candidat[-5] = 1/population#= 200/67000000
	tauxGraves = candidat[-4] = 1/200
	tauxRemis = candidat[-3] = 1/10
	tauxRemisGraves = candidat[-2]
	tauxMortalite = candidat[-1]
	
	# Paramètres fixes
	#tauxRemis = 1/30
	#tauxGraves = 0.07
	#tauxInfecteInitial = 30/67000000

	I=[tauxInfecteInitial]
	
	G=[data_objectif[2][0]]
	R=[data_objectif[3][0]]
	M=[data_objectif[4][0]]

	S=[1 - I[0] - G[0] - R[0] - M[0]]

	liste_taux = []
	i = 0
	for periode in periodes_confinement:
		liste_taux = liste_taux + [candidat[i]] * periode
		i += 1

	SIGRM_increment(S, I, G, R, M, liste_taux, tauxRemis, tauxGraves, tauxRemisGraves, tauxMortalite)
	candidat_data = np.array([S, I, G, R, M])

	if plot:
		fig, ax = plt.subplots(figsize=(12,7))
		plt.plot(data_objectif[2]*population, label="Graves (données)")
		plt.plot(np.array(G)*population, label="Graves (estimation")
		plt.plot(data_objectif[4]*population, label="Morts (données)")
		plt.plot(np.array(M)*population, label="Morts (estimation)")
		plt.legend()
		plt.savefig(str(plot) +".jpg")

	fitness= np.linalg.norm(np.array([candidat_data[2],candidat_data[4]] ) - np.array([data_objectif[2],data_objectif[4]]) ),
	if altered :
		return fitness[0], candidat.tolist()
	else :
		return fitness

def SIRHGM_increment(S, I, R, H, G, M, liste_taux_SI, taux_IR, taux_IH, taux_IG, taux_HR, taux_HG, taux_GR, taux_GH, taux_GM, check=False):

	for taux_SI in liste_taux_SI:


		nouveaux_SI = taux_SI * S[-1] * I[-1]	

		nouveaux_IR = taux_IR * I[-1]
		nouveaux_IH = taux_IH * I[-1]
		nouveaux_IG = taux_IG * I[-1]

		nouveaux_HR = taux_HR * H[-1]
		nouveaux_HG = taux_HG * H[-1]

		nouveaux_GR = taux_GR * G[-1]
		nouveaux_GH = taux_GH * G[-1]
		nouveaux_GM = taux_GM * G[-1]
		
		S.append(S[-1] - nouveaux_SI) 
		I.append(I[-1] + nouveaux_SI - nouveaux_IR - nouveaux_IH - nouveaux_IG)
		R.append(R[-1] + nouveaux_HR + nouveaux_GR + nouveaux_IR)
		H.append(H[-1] + nouveaux_IH + nouveaux_GH - nouveaux_HR - nouveaux_HG)
		G.append(G[-1] + nouveaux_IG + nouveaux_HG - nouveaux_GR - nouveaux_GH - nouveaux_GM)
		M.append(M[-1] + nouveaux_GM)
		if check:
			assert(round(S[-1]+I[-1]+R[-1]+H[-1]+G[-1]+M[-1],1) == 1) # la somme fait toujours 1
	return None

def eval_SIRHGM(candidat, data_objectif, periodes_confinement, population=1, plot=False, check=False, description=False, altered=False):
	candidat = np.absolute(candidat)

	taux_I0 = candidat[-9] = 10/3000000
	taux_IR = candidat[-8] 
	taux_IH = candidat[-7] 
	taux_IG = candidat[-6] 
	taux_HR = candidat[-5]
	taux_HG = candidat[-4]
	taux_GR = candidat[-3]
	taux_GH = candidat[-2] = 0
	taux_GM = candidat[-1]

	S=[1 - taux_I0]
	I=[taux_I0]
	R=[0.0]
	H=[0.0]	
	G=[0.0]
	M=[0.0]
	liste_taux_SI = []
	i = 0
	for periode in periodes_confinement:
		liste_taux_SI = liste_taux_SI + [candidat[i]] * periode
		i += 1

	SIRHGM_increment(S, I, R, H, G, M, liste_taux_SI, taux_IR, taux_IH, taux_IG, taux_HR, taux_HG, taux_GR, taux_GH, taux_GM, check)
	candidat_data = np.array([S, I, R, H, G, M])
	if plot:
		fig, ax = plt.subplots(figsize=(12,7))
		plt.plot((data_objectif[3] + data_objectif[4])*3000000, label="Hospitalisés (données)", color='blue')
		plt.plot(np.array(H)*3000000 + np.array(G)*3000000, label="Hospitalisés (estimation)", color='blue', linestyle = ":")
		plt.plot(data_objectif[4]*3000000, label="Réanimation (données)", color='red')
		plt.plot(np.array(G)*3000000, label="Réanimation (estimation)", color='red', linestyle = ":")
		plt.plot(data_objectif[5]*3000000, label="Décès (données)", color='black')
		plt.plot(np.array(M)*3000000, label="Décès (estimation)", color='black', linestyle = ":")
		plt.legend()
		plt.savefig(str(plot) +".jpg")

	if description:
		for i in range(len(periodes_confinement)):
			print("R0 période " + str(i) + " : " + str(candidat[i]/(taux_IR + taux_IH + taux_IG)))
		print(str(taux_I0 * 3000000) + " infectés initiaux")
		print("Taux I->R : " + str(round(taux_IR * 100, 2)) + " pour cent.")
		print("Taux I->H : " + str(round(taux_IH * 100, 2)) + " pour cent.")
		print("Taux I->G : " + str(round(taux_IG * 100, 2)) + " pour cent.")
		print("Taux H->R : " + str(round(taux_HR * 100, 2)) + " pour cent.")
		print("Taux H->G : " + str(round(taux_HG * 100, 2)) + " pour cent.")
		print("Taux G->R : " + str(round(taux_GR * 100, 2)) + " pour cent.")
		print("Taux G->H : " + str(round(taux_GH * 100, 2)) + " pour cent.")
		print("Taux G->M : " + str(round(taux_GM * 100, 2)) + " pour cent.")
		print(str(round((I[-1] + R[-1] + H[-1] + G[-1] + M[-1])*100, 2)) + " pour cent de la population a été infectée.")

	fitness = np.linalg.norm(np.array([candidat_data[3], candidat_data[4], candidat_data[5]] ) - np.array([data_objectif[3], data_objectif[4], data_objectif[5]]) ),

	if altered :
		return fitness[0], candidat.tolist()
	else :
		return fitness

def SIR2HGM_increment(S, I, R, R2, H, G, M, liste_taux_SI, taux_IR2, taux_IH, taux_IG, taux_HR, taux_HG, taux_GR, taux_GH, taux_GM, check=False):

	for taux_SI in liste_taux_SI:

		nouveaux_SI = taux_SI * S[-1] * I[-1]	

		nouveaux_IR2 = taux_IR2 * I[-1]
		nouveaux_IH = taux_IH * I[-1]
		nouveaux_IG = taux_IG * I[-1]

		nouveaux_HR = taux_HR * H[-1]
		nouveaux_HG = taux_HG * H[-1]

		nouveaux_GR = taux_GR * G[-1]
		nouveaux_GH = taux_GH * G[-1]
		nouveaux_GM = taux_GM * G[-1]
		
		S.append(S[-1] - nouveaux_SI) 
		I.append(I[-1] + nouveaux_SI - nouveaux_IR2 - nouveaux_IH - nouveaux_IG)
		R.append(R[-1] + nouveaux_HR + nouveaux_GR)
		R2.append(R2[-1] + nouveaux_IR2)
		H.append(H[-1] + nouveaux_IH + nouveaux_GH - nouveaux_HR - nouveaux_HG)
		G.append(G[-1] + nouveaux_IG + nouveaux_HG - nouveaux_GR - nouveaux_GH - nouveaux_GM)
		M.append(M[-1] + nouveaux_GM)
		if check:
			assert(round(S[-1]+I[-1]+R[-1]+R2[-1]+H[-1]+G[-1]+M[-1],1) == 1) # la somme fait toujours 1
	return None

def eval_SIR2HGM(candidat, data_objectif, periodes_confinement, population=1, plot=False, check=False, description=False, altered=False):
	candidat = np.absolute(candidat)

	taux_I0 = candidat[-9] = 3/67000000
	taux_IR2 = candidat[-8] 
	taux_IH = candidat[-7] 
	taux_IG = candidat[-6] 
	taux_HR = candidat[-5]
	taux_HG = candidat[-4]
	taux_GR = candidat[-3]
	taux_GH = candidat[-2] = 0
	taux_GM = candidat[-1]

	S=[1 - taux_I0]
	I=[taux_I0]
	R=[0.0]
	R2=[0.0]
	H=[0.0]	
	G=[0.0]
	M=[0.0]
	liste_taux_SI = []
	i = 0
	for periode in periodes_confinement:
		liste_taux_SI = liste_taux_SI + [candidat[i]] * periode
		i += 1

	SIR2HGM_increment(S, I, R, R2, H, G, M, liste_taux_SI, taux_IR2, taux_IH, taux_IG, taux_HR, taux_HG, taux_GR, taux_GH, taux_GM, check)
	candidat_data = np.array([S, I, R, R2, H, G, M])
	if plot:
		fig, ax = plt.subplots(figsize=(12,7))
		plt.plot(data_objectif[2]*67000000, label="Remis (données)", color='green')
		plt.plot(np.array(R)*67000000, label="Remis (estimation)", color='green', linestyle = ":")
		plt.plot((data_objectif[4] + data_objectif[5])*67000000, label="Hospitalisés (données)", color='blue')
		plt.plot(np.array(H)*67000000 + np.array(G)*67000000, label="Hospitalisés (estimation)", color='blue', linestyle = ":")
		plt.plot(data_objectif[5]*67000000, label="Réanimation (données)", color='red')
		plt.plot(np.array(G)*67000000, label="Réanimation (estimation)", color='red', linestyle = ":")
		plt.plot(data_objectif[6]*67000000, label="Décès (données)", color='black')
		plt.plot(np.array(M)*67000000, label="Décès (estimation)", color='black', linestyle = ":")
		plt.legend()
		plt.savefig(str(plot) +".jpg")

	if description:
		for i in range(len(periodes_confinement)):
			print("R0 période " + str(i) + " : " + str(candidat[i]/(taux_IR2 + taux_IH + taux_IG)))
		print(str(taux_I0 * 67000000) + " infectés initiaux")
		print("Taux I->R2 : " + str(round(taux_IR2 * 100, 2)) + " pour cent.")
		print("Taux I->H : " + str(round(taux_IH * 100, 2)) + " pour cent.")
		print("Taux I->G : " + str(round(taux_IG * 100, 2)) + " pour cent.")
		print("Taux H->R : " + str(round(taux_HR * 100, 2)) + " pour cent.")
		print("Taux H->G : " + str(round(taux_HG * 100, 2)) + " pour cent.")
		print("Taux G->R : " + str(round(taux_GR * 100, 2)) + " pour cent.")
		print("Taux G->H : " + str(round(taux_GH * 100, 2)) + " pour cent.")
		print("Taux G->M : " + str(round(taux_GM * 100, 2)) + " pour cent.")
		print(str(round((I[-1] + R[-1] + R2[-1] + H[-1] + G[-1] + M[-1])*100, 2)) + " pour cent de la population a été infectée.")

	fitness = np.linalg.norm(np.array([candidat_data[2], candidat_data[4], candidat_data[5], candidat_data[6]] ) - np.array([data_objectif[2], data_objectif[4], data_objectif[5], data_objectif[6]]) ),

	if altered :
		return fitness[0], candidat.tolist()
	else :
		return fitness