"""This module contains the AcControlOptimizer class."""

import random

from typing import List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from gene_change import GeneChange
from gene_translation import GeneTranslation
from sa_method import SA
from selection_method import SelectionMethod
from config.project_setting import temperature_predict_settings, bill_contract_price
from src.service.temperature_prediction.prediction import Prediction


class AcControlOptimizer:
    """AcControlOptimizer class.
    args:
        population_size: 基因池大小
        number_of_features: 基因長度(有多少個控制項)
        future_step: 預測未來幾個時間點
        crossover_rate: 交配率
        mutation_rate: 突變率
        iteration: 迭代次數
    """

    def __init__(self,
                population_size: int,
                number_of_features: int,
                future_step: int,
                crossover_rate: float,
                mutation_rate: float,
                iteration: int
            ) -> None:
        self.population_size = population_size
        self.number_of_features = number_of_features
        self.future_step = future_step
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.iteration = iteration

        self.status_columns = temperature_predict_settings.status_columns
        self.ac_temperature_action = temperature_predict_settings.temperature_setting_list
        self.ac_status_action = temperature_predict_settings.ac_status_list
        self.ac_mode_action = temperature_predict_settings.ac_mode_list
        self.ac_wind_action = temperature_predict_settings.wind_setting_list
        self.control_ac_mode_action = temperature_predict_settings.control_ac_mode_list
        self.control_ac_wind_action = temperature_predict_settings.control_wind_setting_list
        self.time_weight = temperature_predict_settings.time_weight
        self.heat_weight = temperature_predict_settings.heat_weight
        self.cold_weight = temperature_predict_settings.cold_weight
        self.window_size = temperature_predict_settings.window_size
        self.summer_month = bill_contract_price.summer_month

    def run(self,
            ac_id: str,
            bill_contract_type: str,
            opend_time: str,
            close_time: str,
            now: datetime = None,
            target_temperature_high: float = 24.5,
            target_temperature_low: float = 18,
            ac_data: pd.DataFrame = None,
            ac_meter_data: pd.DataFrame = None
            ) -> List[int]:
        """Run the optimization process.
            ac_id: 空調id
            bill_contract_type: 電費合約類型
            opend_time: 門市開門的營業時間 11:30
            close_time: 門市關門的營業時間 21:00
            Returns:
                List[int]: Best solution.
        """
        try:
            # start_time = time.time()

            opend_time =  datetime.strptime(opend_time, "%H:%M").time()
            close_time = datetime.strptime(close_time, "%H:%M").time()

            bill_contract = bill_contract_price.__dict__[bill_contract_type]
            prex = bill_contract_type.split("_")[0]

            prediction = Prediction(ac_id = ac_id, current = now)


            # Import real-time data
            if ac_data is None or ac_data.empty or ac_meter_data is None or ac_meter_data.empty:
                ac_data, ac_meter_data = prediction.get_current_raw_data()
            _, _, real_time_data = prediction.get_inference_data(ac_data, ac_meter_data)

            last_ac_status =real_time_data["ac_status"].iloc[-1]
            last_status = real_time_data.iloc[-self.window_size:][self.status_columns].values.flatten().tolist()
            last_data_time = real_time_data.index[-1]
            hour_list, is_open , ac_opend = self.get_hour_is_open_ac_open_list(opend_time, close_time, last_data_time, 15)
            price_list = self.get_price_list(bill_contract, prex, last_data_time, hour_list)

            actions = []
            actions.append(self.ac_temperature_action)
            actions.append(self.ac_status_action)
            actions.append(self.control_ac_mode_action)
            actions.append(self.control_ac_wind_action)
            accuracy = []
            accuracy.append(len(self.ac_temperature_action))
            accuracy.append(len(self.ac_status_action))
            accuracy.append(len(self.control_ac_mode_action))
            accuracy.append(len(self.control_ac_wind_action))

            #GA start
            gene_translation = GeneTranslation(
                    pop_size = self.population_size,
                    number_of_features = self.number_of_features,
                    future_step = self.future_step,
                    accuracy = accuracy
                )
            
            gene_change = GeneChange(accuracy, self.future_step)
            parent_pop = gene_translation.create_encoded_pop()#取得基因
            parent_pop = gene_change.repair_ac_status(parent_pop, ac_opend, last_ac_status)
            phenotypes = gene_translation.decode_chrom(parent_pop, actions)#取得表現型
            fit_value_parent = self.get_fit_value(prediction, is_open, price_list, phenotypes, last_status, target_temperature_high, target_temperature_low)
            
            
            results = [[]]
            plotbest = []
            plotmean = []
            for i in range(self.iteration):
                pop = parent_pop.copy()
                # pop = gene_change.crossover(pop, self.crossover_rate)

                # pop = gene_change.mutation(pop, self.mutation_rate)
                pop = gene_translation.create_encoded_pop()#取得基因

                pop = gene_change.repair_ac_status(pop, ac_opend, last_ac_status)

                phenotypes = gene_translation.decode_chrom(pop, actions)#取得表現型

                fit_value = self.get_fit_value(prediction, is_open, price_list, phenotypes, last_status, target_temperature_high, target_temperature_low)

                pop.extend(parent_pop)
                
                fit_value.extend(fit_value_parent)


                selection_method = SelectionMethod(self.population_size)
                
                
                best_individual, best_fit = selection_method.get_best_individual(pop, fit_value)

                results.append([best_fit, best_individual])
                
                plotbest.append(best_fit)
                plotmean.append(np.mean(fit_value))

                # print(np.mean(fit_value))
                

                parent_pop, fit_value_parent = selection_method.selection(fit_value, parent_pop, fit_value_parent, pop)


                parent_pop.append(best_individual) #加上最佳解
                fit_value_parent.append(best_fit) #加上最佳解


            results = results[1:]
            results.sort(key = lambda x: x[0])
            #GA end

            #SA start
            sa = SA()
            temperature = results[0][0] + 1
            # 原始obj為使用list形式的pop計算，SA用單一解出發
            cur_state = []
            cur_state.append(results[0][1][:])  #繼承GA的最佳解
            cur_fit = results[0][0]
            best_state = cur_state[0]
            best_fit = results[0][0]
            #solution_temp = solution[:]
            # solution.pop()
            anneal_factor = 0.85

            k=0
            while k < 50:
                #跨出新解

                next_state = sa.anneal_stride(cur_state)
                next_state = gene_change.repair_ac_status(next_state, ac_opend, last_ac_status)
                #計算finess

                phenotypes = gene_translation.decode_chrom(next_state, actions)#取得表現型

                anneal_fit_value = self.get_fit_value(prediction, is_open, price_list, phenotypes, last_status, target_temperature_high, target_temperature_low)
                

                #計算機率&選擇
                delta_c = cur_fit - anneal_fit_value[0]  
                """
                print(str(k) + "  /cur_fit: " + str(cur_fit) )
                print("   /new: "+  str(anneal_fit_value[0])  )
                print("   /delta c: "+ str(delta_c))
                print("   /temp: "+ str(temperature)       )
                print("   /prob: "+str(anneal_prob(delta_c, temperature)))
                """
                if sa.anneal_prob(delta_c, temperature) > random.random() : #如果通過Metropolis prob檢驗
                    #更新fit value & solution    
                    cur_fit = anneal_fit_value[0]
                    cur_state.pop()
                    cur_state= cur_state + next_state

                if best_fit > anneal_fit_value[0]  :  # 如果有找到更佳解
                    best_fit = anneal_fit_value[0]
                    best_state = next_state[0]
                #print("   / best: "+str(best_fit)+"\n")
                results.append([best_fit, best_state])
                plotbest.append(best_fit)
                plotmean.append(anneal_fit_value[0])
                temperature = max(0.001, temperature * anneal_factor)
                next_state.pop()
                k = k+1

                #SA End


            results.sort(key = lambda x: x[0])
            
            best_individual= results[0][1]
            best_solution = self.decode_best_solution(best_individual, actions)
            # end_time = time.time()
            return best_solution#, plotbest, plotmean
        finally:
            prediction.release_resources()
            del prediction, gene_translation, gene_change, selection_method, sa, results, pop, parent_pop, fit_value, fit_value_parent, best_individual, best_fit, next_state, cur_state, anneal_fit_value, phenotypes, hour_list, is_open, ac_opend, price_list, actions, accuracy, last_ac_status, last_status, real_time_data, ac_data, ac_meter_data
            
    

    """
    GA 評估 : 設計損失函數評估解的好壞 
    """
    def get_fit_value(self, prediction: Prediction, is_open: list, price_list: list, phenotypes: np.array, last_status: pd.DataFrame, target_temperature_high: float, target_temperature_low: float) -> list:
        
        fit_value = []
        for phenotype in phenotypes:
            new_setting = []
            for step in phenotype.T:
                ac_mode = step[-2]
                ac_wind = step[-1]
                ac_mode_status = self.to_onehot(ac_mode, self.ac_mode_action)
                ac_wind_status = self.to_onehot(ac_wind, self.ac_wind_action)
                step = step[:-2].tolist()
                step.extend(ac_mode_status)
                step.extend(ac_wind_status)
                new_setting.extend(step)
            temperature_list= []
            kwh_list = []
            status = last_status.copy()
            
            status.extend(new_setting)

            temperature_list, kwh_list = prediction.make_prediction(model_input=status)

            hot_value = 0
            cold_value = 0
            total_price = 0
            for time, opened in enumerate(is_open):
                if opened != 0:
                    hot_value += max(0, temperature_list[time] - target_temperature_high)*self.time_weight[time]
                    cold_value += max(0, target_temperature_low - temperature_list[time])*self.time_weight[time]
                    total_price += kwh_list[time]*price_list[time]*self.time_weight[time]

            fit_value.append((total_price + hot_value*self.heat_weight + cold_value*self.cold_weight))
        return fit_value

    def get_hour_is_open_ac_open_list(self,opend_time: datetime, close_time: datetime, last_data_time: datetime, time_range: int = 15)-> Tuple[list, list, list]:
        hour_list = []
        is_open = []
        ac_open = []
        current = datetime.now()
        current_datetime_with_close_time = datetime.combine(current.date(), close_time)
        ac_close_time = (current_datetime_with_close_time - timedelta(hours=1)).time()
        current_datetime_with_opend_time = datetime.combine(current.date(), opend_time)
        ac_open_time = (current_datetime_with_opend_time - timedelta(hours=3)).time()

        for time in range(1,self.future_step+1):
            next_time = last_data_time + timedelta(minutes=time_range*time)
            if opend_time <= next_time.time() < close_time:
                is_open.append(1)
            else:
                is_open.append(0)
            
            if ac_open_time <= next_time.time() < ac_close_time:
                ac_open.append(1)
            else:
                ac_open.append(0)

            hour_list.append(next_time.hour)
        return hour_list, is_open, ac_open


    def get_hour_price_dict(self, bill_price: dict)-> dict:
        hour_price_pair = bill_price['hour_price_pair']
        hour_price_dict = {}
        for hour in hour_price_pair:
            hour_list = list(range(int(hour.split("_")[0]), int(hour.split("_")[1])))
            hour_dict = {num: hour_price_pair[hour] for num in hour_list}
            hour_price_dict.update(hour_dict)
        return hour_price_dict
    
    def get_price_list(self, bill_contract, prex, last_data_time, hour_list: list)-> list:
        weekday = last_data_time.weekday()
        month = last_data_time.month
        bill_price = self.get_bill_price(bill_contract, prex, weekday, month)
        hour_price_dict = self.get_hour_price_dict(bill_price)
        price_list = []
        for hour in hour_list:
            price_list.append(hour_price_dict[hour])
        return price_list

    def get_bill_price(self, bill_contract, prex, weekday, month)-> dict:
        if prex == "usage":
            bill_price = {"hour_price_pair": {
                            "0_24": 2.53
                        }
                    }
        elif month in self.summer_month: #summer
            if weekday < 5: # weekday
                bill_price = bill_contract.summer_weekday   
            else:
                if prex == "simple":
                    bill_price = bill_contract.summer_weekend
                else :# prex = "normal"
                    if weekday == 5: #sat
                        bill_price = bill_contract.summer_sat
                    else:
                        bill_price = bill_contract.summer_sun              
        else:
            if weekday < 5: # weekday
                bill_price = bill_contract.non_summer_weekday
            else:
                if prex == "simple":
                    bill_price = bill_contract.non_summer_weekend
                else: # prex = "normal"
                    if weekday == 5: #sat
                        bill_price = bill_contract.non_summer_sat
                    else:
                        bill_price = bill_contract.non_summer_sun
        return bill_price
    
    def decode_best_solution(self, best_solution: list, actions: list)-> list:
        phenotype = []
        for action in range(len(actions)):
            action_array = np.array(actions[action], dtype=float)
            phenotype_array = np.asarray(best_solution[action] * action_array)
            action_phenotype = [np.sum(array) for array in phenotype_array]     
            phenotype.append(action_phenotype)
        return phenotype

    @staticmethod
    def to_onehot(setting_value, setting_list)-> list:
        setting_status = [0]*len(setting_list)
        index = [index for index, value in enumerate(setting_list) if int(value) == int(setting_value)]
        setting_status[index[0]] = 1
        return setting_status