__author__ = '이다영, 박혜진'

import os

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pathlib
import pickle
import time

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint
from sc2.position import Point2, Point3
from enum import Enum
from random import *

from .consts import CommandType, ProductStrategy, NukeStrategy

nest_asyncio.apply()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
         #5가 state 개수, 12가 유닛종류(economy)
        self.fc1 = nn.Linear(6, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.vf = nn.Linear(64, 1)
        self.product_head = nn.Linear(64, len(ProductStrategy))
        self.nuke_head = nn.Linear(64, len(NukeStrategy))
        #self.mule_head = nn.Linear(64, len(MuleStrategy))


    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        value = self.vf(x)
        product_logp = torch.log_softmax(self.product_head(x), -1)
        nuke_logp = torch.log_softmax(self.nuke_head(x), -1)
        #mule_logp = torch.log_softmax(self.mule_head(x), -1)
        bz = x.shape[0]
        logp = (product_logp.view(bz, -1, 1) + nuke_logp.view(bz, 1, -1)).view(bz, -1)
        #logp = (product_logp.view(bz, 1, -1, -1) + nuke_logp.view(bz, -1, 1, -1) + mule_logp.view(bz, -1, -1, 1)).view(bz, -1)
        return value, logp

    
class NukeManager(object):
    """
    사령부 핵 담당 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.pos=0
        self.ghost_pos1_x=32.5
        self.ghost_pos2_x=95.5
        self.ghost_pos=0
        self.enemy_pos=0

    def reset(self):
        if self.bot.enemy_start_locations[0].x == 95.5:
            self.ghost_pos = self.ghost_pos1_x
            self.enemy_pos= self.ghost_pos2_x
        else :
            self.ghost_pos = self.ghost_pos2_x
            self.enemy_pos= self.ghost_pos1_x
        
        
        

    # 마지막 명령이 발행된지 10초가 넘었으면 리워드 -1?
    # 택틱마다 일정시간 지나도 명령 발행 안되면 리워드 -1 

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ghosts = self.bot.units(UnitTypeId.GHOST)
        
        if ghosts.amount == 0:
            actions.append(cc.train(UnitTypeId.GHOST))

        if ghosts.amount > 0:
            ghost = ghosts.first
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(cc(AbilityId.BUILD_NUKE))
                #print(ghosts.first.position)

            if ghost.distance_to(Point2((self.ghost_pos,55))) > 3 and self.pos == 0:
                actions.append(ghost.move(Point2((self.ghost_pos,55))))
                self.pos=1

            if ghost.distance_to(Point2((self.ghost_pos,55))) == 0 and self.pos == 1:
                #print(ghost.position)
                actions.append(ghosts.first.move(Point2((self.enemy_pos,55))))
                self.pos=2

            if ghost.distance_to(Point2((self.enemy_pos,55))) < 3:
                #print(ghost.position)
                self.pos=3

            if self.pos==3 :
                ghost_abilities = await self.bot.get_available_abilities(ghosts.first)
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghosts.first.is_idle:
                    # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                    actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    actions.append(ghosts.first(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_start_locations[0]))

        return actions


class ReconManager(object):
    """
    정찰부대 운용을 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.perimeter_radious = 10
        self.pos=0
        self.pos1_x=32.5
        self.pos2_x=95.5

    def reset(self):
        if self.bot.enemy_start_locations[0].x == 95.5:
            self.pos = self.pos1_x
        else :
            self.pos = self.pos2_x

    # def position(self):
        
    async def step(self):
        actions = list()

        for unit in self.bot.units:
            if unit.tag in self.bot.reconArray:
                # 근처에 적들이 있는지 파악
                actions.append(unit.move(Point2((self.pos,40))))
                threaten = self.bot.known_enemy_units.closer_than(
                        self.perimeter_radious, unit.position)
                
                if unit.type_id == UnitTypeId.RAVEN:
                    if unit.health_percentage > 0.8 and unit.energy >= 50:
                        #print("유닛오더? ",unit.orders)
                        if threaten.amount > 0: # 근처에 적이 하나라도 있으면
                            alert = 1
                            if unit.orders and unit.orders[0].ability.id != AbilityId.RAVENBUILD_AUTOTURRET:
                                closest_threat = threaten.closest_to(unit.position)
                                pos = unit.position.towards(closest_threat.position, 5)
                                pos = await self.bot.find_placement(
                                    UnitTypeId.AUTOTURRET, pos)
                                order = unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos)
                                actions.append(order)
                    '''else:
                        if unit.distance_to(self.target) > 5:
                            order = unit.move(self.target)
                            actions.append(order)'''

                elif unit.type_id == UnitTypeId.MARINE:
                    if self.bot.known_enemy_units.exists:
                        enemy_unit = self.bot.known_enemy_units.closest_to(unit)
                        actions.append(unit.attack(enemy_unit))

        return actions


class CombatManager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):
        self.evoked = dict()  

    async def step(self):
        actions = list()

        ##-----변수, 그룹 등 선언-----
        '''combat_tag = self.bot.units.filter(
            lambda unit: unit.tag in self.bot.combatArray
        )'''
        #combatArray에 있는 유닛들을 units로
        combat_units = self.bot.units.tags_in(self.bot.combatArray)
        #combat_units = self.bot.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.RAVEN, UnitTypeId.GHOST, UnitTypeId.MULE])
        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치
        cc_abilities = await self.bot.get_available_abilities(cc)
        mule = self.bot.units(UnitTypeId.MULE)
        #pos = cc.position.towards(enemy_cc.position, -5)
        #방어시 집결 위치
        defense_position = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        #다친 기계 유닛
        wounded_units = self.bot.units.filter(
            lambda u: u.is_mechanical and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색

        
        #Mule 기계유닛 힐
        if wounded_units.exists or cc.health_percentage < 0.8:
            wounded_unit = wounded_units.closest_to(cc)
            if mule.amount == 0:
                #print("뮬없음")
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, wounded_unit.position))
                    #print("뮬생산")
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, wounded_unit))
                #print("힐")
        else: #택틱이 뮬인데 치료할거리가 없을경우->그럴수가 있나?
            if mule.amount > 0:
                mule_unit=mule.random
                actions.append(mule_unit.move(combat_units.center))
                #print("이동")

        
        #
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            if unit in combat_units:
                    ##-----타겟 설정-----
                enemy_unit = self.bot.enemy_start_locations[0]
                if self.bot.known_enemy_units.exists:
                    enemy_unit = self.bot.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

                # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
                if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                    target = enemy_cc
                else:
                    target = enemy_unit

                ##-----유닛 스킬-----

                ##-----전투 유닛 전체-----
                if unit.type_id is not UnitTypeId.MEDIVAC:
                    #print(combat_units.amount)
                    #if len(self.bot.combatArray) > 20 and self.bot.units.of_type([UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK]).amount > 1:
                    if self.bot.combat_strategy is CombatStrategy.OFFENSE: #0=OFFENSE, 2=WAIT
                        #unit.move(Point2((self.pos,60)))
                        # 전투가능한 유닛 수가 20을 넘으면 + 탱크 2대 이상 적 본진으로 공격              
                        actions.append(unit.attack(target))
                    elif self.bot.combat_strategy is CombatStrategy.DEFENSE: #1=DEFENSE
                        # 적 사령부 방향에 유닛 집결
                        target = defense_position
                        actions.append(unit.attack(target))

                    ##-----해병과 불곰-----
                    if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                        #OFFENSE
                        if self.bot.combat_strategy is CombatStrategy.OFFENSE and unit.distance_to(target) < 15:
                            # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                            if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                                # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                                if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                    # 1초 이전에 스팀팩을 사용한 적이 없음
                                    actions.append(unit(AbilityId.EFFECT_STIM))
                                    self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time
                        
                        
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes

                    ##-----토르-----
                    if unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP):
                        """
                        if unit.health_percentage < 0.5:
                            if mule.amount == 0:
                            #print("뮬없음")
                                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                                    # 지게로봇 생산가능하면 생산
                                    actions.append(cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, unit.position))
                                    #print("뮬생산")
                            elif mule.amount > 0:
                                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, unit))
                                #print("힐")
                            else: #뮬이 치료할거리가 없을경우
                                if mule.amount > 0:
                                    mule_unit=mule.random
                                    actions.append(mule_unit.move(combat_units.center))"""
                        
                        if self.bot.known_enemy_units.exists:
                            #적에 전투순양함이 있으면 전투순양함에게 한방에 큰 딜 넣는게 우선
                            if self.bot.known_enemy_units.of_type(UnitTypeId.BATTLECRUISER).exists:
                                target = self.bot.known_enemy_units.of_type(UnitTypeId.BATTLECRUISER).closest_to(unit)
                                #print("토르 타겟: ", target)
                                if unit.type_id is UnitTypeId.THOR:
                                    order = unit(AbilityId.MORPH_THORHIGHIMPACTMODE)
                                    actions.append(order)
                            #전투순양함 없으면 광역딜
                            else:
                                if unit.type_id is UnitTypeId.THORAP:
                                    order = unit(AbilityId.MORPH_THOREXPLOSIVEMODE)
                                    actions.append(order)
                        

                    ##-----토르-----
                    if unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP):
                        """
                        if unit.health_percentage < 0.5:
                            if mule.amount == 0:
                            #print("뮬없음")
                                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                                    # 지게로봇 생산가능하면 생산
                                    actions.append(cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, unit.position))
                                    #print("뮬생산")
                            elif mule.amount > 0:
                                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, unit))
                                #print("힐")
                            else: #뮬이 치료할거리가 없을경우
                                if mule.amount > 0:
                                    mule_unit=mule.random
                                    actions.append(mule_unit.move(combat_units.center))"""
                        
                        if self.bot.known_enemy_units.exists:
                            #적에 전투순양함이 있으면 전투순양함에게 한방에 큰 딜 넣는게 우선
                            if self.bot.known_enemy_units.of_type(UnitTypeId.BATTLECRUISER).exists:
                                target = self.bot.known_enemy_units.of_type(UnitTypeId.BATTLECRUISER).closest_to(unit)
                                #print("토르 타겟: ", target)
                                if unit.type_id is UnitTypeId.THOR:
                                    order = unit(AbilityId.MORPH_THORHIGHIMPACTMODE)
                                    actions.append(order)
                            #전투순양함 없으면 광역딜
                            else:
                                if unit.type_id is UnitTypeId.THORAP:
                                    order = unit(AbilityId.MORPH_THOREXPLOSIVEMODE)
                                    actions.append(order)
                        


                    '''
                    ##-----탱크-----
                    if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                        #상태에 따라 분류
                        #적과의 거리 멀고 + 전차모드
                        if unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            unit_state = 0
                        #적과의 거리 멀고 + 전차모드x
                        elif unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANK:
                            unit_state = 1
                        #적과의 거리 가깝고 + 전차모드
                        elif unit.distance_to(target) <= 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            unit_state = 2
                        #적과의 거리 가깝고 + 전차모드x
                        elif unit.distance_to(target) <= 10 and unit.type_id == UnitTypeId.SIEGETANK:
                            unit_state = 3
                        else:
                            unit_state = 4

                        #DEFENSE
                        if self.bot.combat_strategy is CombatStrategy.DEFENSE:
                            if unit.distance_to(defense_position) < 5:
                                if unit.type_id == UnitTypeId.SIEGETANK:
                                    order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                    actions.append(order) 
                                    #print("탱크됨")   
                                    
                        #OFFENSE        
                        elif self.bot.combat_strategy is CombatStrategy.OFFENSE:
                            #if unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            #적이랑 멀고 전차모드임->이동->전차모드 해제
                            if unit_state == 0:
                                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                actions.append(order)
                            #적이랑 가깝고 전차아님->공격->전차모드 됨
                            elif unit_state == 3:
                                order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                actions.append(order) 
                            #적이랑 멀고 전차아님(1)->그냥 target으로 가면 됨
                            #적이랑 가깝고 전차임(2)->그냥 target 때리면 됨
                            
                        #WAIT
                        elif self.bot.combat_strategy is CombatStrategy.WAIT:
                            #print("방어위치: ", defense_position)
                            #print("중앙집결: ", combat_units.center)
                            #적이 있으면 치는게 낫나? 아니면 그냥 빼는게 낫나? 탱크는 느려서...
                            #전차모드임->이동(후퇴)->전차모드 해제하고 이동
                            if unit_state == 0 or unit_state == 2:
                                #전차 해제
                                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                actions.append(order)

                            #대기 위치로 이동
                            actions.append(unit.move(wait_position))
                            #대기 위치 근처면 전차됨
                            if unit.distance_to(wait_position) < 3:
                                order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                actions.append(order)   
                        
                        #타겟 우선순위 설정
                        if self.bot.known_enemy_units.exists:
                            #print("탱크 적 있음")
                            #타겟설정: 해병과 불곰 우선
                            if self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).exists:
                                target = self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).closest_to(unit)
                                #print("탱크 타겟설정됨")'''
                            
            
                                


            ##-----비전투 유닛==메디박-----
            """
            if unit.type_id is UnitTypeId.MEDIVAC:
                ##힐을 가장 우선시
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                    #print(wounded_unit.name, "을 회복중")
                elif len(self.bot.combatArray) < 1 :
                    actions.append(unit.move(self.bot.cc - 5))
                else: 
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                    actions.append(unit.move(self.bot.units.closest_to(unit)))
                    #print("대기중")"""
=======
                    actions.append(unit.move(self.bot.units.closest_to(unit)))"""
>>>>>>> Stashed changes
=======
                    actions.append(unit.move(self.bot.units.closest_to(unit)))"""
>>>>>>> Stashed changes

        # -----유닛 명령 생성 끝-----
        return actions


class StepManager(object):
    """
    스텝 레이트 유지를 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.seconds_per_step = 0.35714  # on_step이 호출되는 주기
        self.reset()
    
    def reset(self):
        self.step = -1
        # 마지막으로 on_step이 호출된 게임 시간
        self.last_game_time_step_evoked = 0.0 

    def invalid_step(self):
        """
        너무 빠르게 on_step이 호출되지 않았는지 검사
        """
        elapsed_time = self.bot.time - self.last_game_time_step_evoked
        if elapsed_time < self.seconds_per_step:
            return True
        else:
            # print(C.blue(f'man_step_time: {elapsed_time}'))
            self.step += 1
            self.last_game_time_step_evoked = self.bot.time
            return False


class ProductManager(object):
    """
    어떤 (부대의) 유닛을 생산할 것인지 결정하는 매니저(선택-생산-배치)
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    async def product(self, next_unit):
        """
        다음에 생산할 유닛 return
        """
        """
        각 부대에서 필요한 유닛을 리스트로 bot에게 줌(비율은 각 매니저에서 계산)
        [combat, recon, nuke] 꼴, 매번 갱신
        이걸 보고 이중에서 어떤걸 생산할지 고르고 생산, 배치를 담당하는 매니저
        어떤걸 우선적으로 고를지는 택틱에 따라 결정
        """
        actions = list()
        #print("00000생산요청들어감: ", self.bot.trainOrder)

        #유닛 생산
        if self.bot.can_afford(next_unit):
            #print("00000생산 가능")
            if self.bot.time - self.bot.evoked.get((self.bot.cc.tag, 'train'), 0) > 1.0:
                #print("00000마지막 명령을 발행한지 1초 이상 지났음")
                actions.append(self.bot.cc.train(next_unit))
                #print("00000생산명령-----:", next_unit)
                #self.bot.productorder = self.bot.productorder + 1
                #print("생산명령횟수: ", self.bot.productorder)
                self.bot.evoked[(self.bot.cc.tag, 'train')] = self.bot.time
        
        return actions     
                     




class RatioManager(object):
    """
    부대에 따라 생성할 유닛 비율 결정하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai

    def next_unit_select(self, unit_counts, target_unit_counts):
        """
        비율에 따라 next_unit 리턴
        """
        target_units = np.array(list(target_unit_counts.values()))
        target_unit_ratio = target_units / (target_units.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        next_unit = list(target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
        return next_unit

    def ratio(self):
        """
        부대별 비율에 따라 다음에 생산할 유닛 bot의 next_unit에 저장
        """
        #combat_unit_counts = dict()
        recon_unit_counts = dict()
        #nuke_unit_counts = dict()

        #COMBAT
        if self.bot.vespene > 200:
            #combat_next_unit = UnitTypeId.SIEGETANK
            combat_next_unit = UnitTypeId.THOR
        else:
            combat_next_unit = UnitTypeId.MARINE
        
        #RECON
        self.recon_target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 2,
                UnitTypeId.MARAUDER: 0, 
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN:1,
                UnitTypeId.BATTLECRUISER: 0,
            }
        for unit in self.bot.units:
            if unit.tag in self.bot.reconArray: # 유닛의 태그가 어레이에 포함되어있으면
                recon_unit_counts[unit.type_id] = recon_unit_counts.get(unit.type_id, 0) + 1
        
        recon_next_unit = self.next_unit_select(recon_unit_counts, self.recon_target_unit_counts)

        #self.bot.trainOrder = [combat_next_unit, recon_next_unit, nuke_next_unit]
        trainOrder = [combat_next_unit, recon_next_unit]
        #유닛 결정
        if self.bot.product_strategy == ProductStrategy.ATTACK: #combat생산
            self.bot.next_unit = trainOrder[0]
        elif self.bot.product_strategy == ProductStrategy.RECON: #recon생산
            self.bot.next_unit = trainOrder[1]  
        #print("11111생산요청: ", self.bot.trainOrder)
        
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        
        """
        #if self.bot.tactic_strategy == TacticStrategy.ATTACK or self.bot.tactic_strategy == TacticStrategy.MULE:
        if self.bot.tactic_strategy == TacticStrategy.ATTACK:
            #print("들어옴: 111111")
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 25,
                UnitTypeId.MARAUDER: 0,
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 2,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN: 0,
                UnitTypeId.BATTLECRUISER: 0,
            }
            for unit in self.bot.units:
                if unit.tag in self.bot.combatArray: # 유닛의 태그가 어레이에 포함되어있으면
                    unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
            
        elif self.bot.tactic_strategy == TacticStrategy.RECON:
            #print("들어옴: 2222222")
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 2,
                UnitTypeId.MARAUDER: 0, 
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN:1,
                UnitTypeId.BATTLECRUISER: 0,
            }
            for unit in self.bot.units:
                if unit.tag in self.bot.reconArray: # 유닛의 태그가 어레이에 포함되어있으면
                    unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1

        elif self.bot.tactic_strategy == TacticStrategy.NUKE:
            #print("들어옴: 333333333")
            self.target_unit_counts = {
                UnitTypeId.GHOST: 1,
            }
            for unit in self.bot.units:
                if unit.tag in self.bot.nukeArray: # 유닛의 태그가 어레이에 포함되어있으면
                    unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1

        

        self.evoked = dict()
        
        
        
        ### 여기서 unit_counts에 각각 다른 유닛집단 들어가야하지않나??

        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련

        print("ratio뽑힌애--------: ",next_unit)
        return next_unit"""
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes


class AssignManager(object):
    """
    유닛을 부대에 배치하는 매니저
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai
        

    def reset(self):
        pass

    def assign(self):
        """
        유닛 타입을 보고 부대별 array에 유닛 배치
        """
        units_tag = self.bot.units.tags #전체유닛
        #and연산으로 살아있는 유닛으로만 구성
        self.bot.combatArray = self.bot.combatArray & units_tag 
        self.bot.reconArray = self.bot.reconArray & units_tag
        self.bot.nukeArray = self.bot.nukeArray & units_tag

        #이미 할당된 유닛의 태그 빼고
        units_tag = units_tag - self.bot.combatArray - self.bot.reconArray - self.bot.nukeArray
        
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        #if self.bot.tactic_strategy == TacticStrategy.ATTACK or self.bot.tactic_strategy == TacticStrategy.MULE:
        if self.bot.tactic_strategy == TacticStrategy.ATTACK:
=======
=======
>>>>>>> Stashed changes
        #유닛 타입에 따라 array 배정
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 레콘으로
                self.bot.reconArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 누크로
                self.bot.nukeArray.add(unit.tag)
            elif unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP): #토르는 컴뱃으로
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)
            else: #아무것도 아니여도 일단 컴뱃으로
                self.bot.combatArray.add(unit.tag)

        """
        if self.bot.product_strategy == ProductStrategy.ATTACK :
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
            self.bot.combatArray = self.bot.combatArray | units_tag
        elif self.bot.product_strategy == ProductStrategy.RECON:
            self.bot.reconArray = self.bot.reconArray | units_tag"""
        #elif self.bot.product_strategy == ProductStrategy.NUKE:
            #self.bot.nukeArray = self.bot.nukeArray | units_tag

        #print("assign됨--------")

    def reassign(self):
        """
        이미 배정된 유닛을 다시 배정
        있을리 없는 유닛타입을 옮겨줌
        """
        for tag in self.bot.combatArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 컴뱃태그에 해당하는 유닛
            if unit is None:
                pass
            elif unit.type_id is UnitTypeId.RAVEN: #레이븐은 레콘으로
                self.bot.reconArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.GHOST: #고스트는 누크로
                self.bot.nukeArray.add(unit.tag)

        for tag in self.bot.reconArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 레콘태그에 해당하는 유닛
            if unit is None:
                pass
            #elif unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
            elif unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP):
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.GHOST:
                self.bot.nukeArray.add(unit.tag)

        for tag in self.bot.nukeArray:
            unit = self.bot.units.find_by_tag(tag) #전체유닛중 누크태그에 해당하는 유닛
            if unit is None:
                pass
            #elif unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
            elif unit.type_id in (UnitTypeId.THOR, UnitTypeId.THORAP):
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.RAVEN:
                self.bot.reconArray.add(unit.tag)       

        #print("reassign됨--------")


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    공격 타이밍은 강화학습을 통해 학습
    """
    def __init__(self, step_interval=5.0, host_name='', sock=None):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / 'model.pt'
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
            except Exception as exc:
                import traceback; traceback.print_exc() 

        self.step_manager = StepManager(self)
        self.combat_manager = CombatManager(self)
        self.assign_manager = AssignManager(self)
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)
        self.ratio_manager = RatioManager(self)
        self.product_manager = ProductManager(self)
        #부대별 유닛 array
        self.combatArray = set()
        self.reconArray = set()
        self.nukeArray = set()

        #self.trainOrder=list()
        self.next_unit = UnitTypeId.MARINE
        
        #self.nukeGo = 0 #핵 쏜 횟수-bigDamage있으면 일단 필요없음 그냥 확인용으로 남겨둠
        #self.previous_Damage = 0 #이전 틱의 토탈대미지(아래를 계산하기 위한 기록용, 매 틱 갱신)
        #self.bigDamage = 0 #한번에 500이상의 큰 대미지가 들어간 횟수(핵이 500이상)
        
        #self.productorder = 0 #생산명령 들어간 횟수
        #self.productdone = 0 #생산명령 수행 횟수
        self.productIng = 0 #생산명령들어가면 1, 처리되면 0

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.product_strategy = ProductStrategy.MARINE.value
        self.nuke_strategy = 1 #0,1
        #self.trainOrder = [UnitTypeId.MARINE, UnitTypeId.MARINE, None] #Combat,Recon,Nuke
        self.trainOrder = [UnitTypeId.MARINE, UnitTypeId.MARINE] #Combat,Recon,Nuke
        self.evoked = dict() #생산명령 시간 체크

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        self.step_manager.reset()
        self.combat_manager.reset()
        self.assign_manager.reset()
        self.nuke_manager.reset()
        self.recon_manager.reset()
        self.assign_manager.assign()


        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌
        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록
        print("1111택틱: ", self.tactic_strategy)

        if self.time - self.last_step_time >= self.step_interval:
            #택틱 변경
            self.product_strategy, self.nuke_strategy = self.set_strategy()
            self.last_step_time = self.time
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            print("-------택틱: ", self.tactic_strategy)
            print("-------컴뱃: ", self.combat_strategy)
=======
            print("-------택틱: ", self.product_strategy)
>>>>>>> Stashed changes
=======
            print("-------택틱: ", self.product_strategy)
>>>>>>> Stashed changes

            self.assign_manager.reassign() #이상하게 배치된 경우 있으면 제배치

            """
            if self.state.score.total_damage_dealt_life - self.previous_Damage > 500: #한번에 500이상의 대미지를 주었다면
                self.bigDamage += 1
                print("한방딜 ㄱ: ", self.bigDamage, "딜량: ", self.state.score.total_damage_dealt_life - self.previous_Damage)
            self.previous_Damage = self.state.score.total_damage_dealt_life #갱신 """
        

        #생산 명령이 처리되었다면
        if self.productIng == 0: 
            self.ratio_manager.ratio() #next_unit 바꿔주고

            #생산
            actions += await self.product_manager.product(self.next_unit) #생산
            self.productIng = 1 #생산명령 들어갔다고 바꿔줌

        #생산 명령이 들어갔다면
        elif self.productIng == 1:
            #대기
            await self.on_unit_created(self.next_unit) #명령넣은게 생산될때까지 기다림
            #self.productdone = self.productdone + 1
            #print("생산명령 수행횟수: ", self.productdone) #엄청큼 문제있음
            #print(self.next_unit)

            #배치
            self.assign_manager.assign()
            self.productIng = 0 #생산명령 수행했다고 바꿔줌 


        
        #actions += await self.attack_team_manager.step()
        
        #actions += await self.product_manager.product(self.next_unit) #생산
        actions += await self.recon_manager.step() 
        actions += await self.combat_manager.step()   
        actions += await self.nuke_manager.step()
        
        ## -----명령 실행-----
        await self.do_actions(actions)
        #print(self.on_unit_created)
        
        #print("한거: ",actions)



    def set_strategy(self):
        #
        # 특징 추출
        #
        state = np.zeros(6, dtype=np.float32)
        state[0] = self.cc.health_percentage
        state[1] = min(1.0, self.minerals / 1000)
        state[2] = min(1.0, self.vespene / 1000)
        state[3] = min(1.0, self.time / 360)
        state[4] = min(1.0, self.state.score.total_damage_dealt_life / 2500)
        state[5] = len(self.combatArray) #combat 부대의 유닛 수 - combat결정용 state
        #state[6] = self.trainOrder #각 부대별 생산요청 유닛 - product 결정용 - 리스트라 안됨
        state = state.reshape(1, -1)

        # NN
        data = [
            CommandType.STATE,
            pickle.dumps(self.game_id),
            pickle.dumps(state.shape),
            state,
        ]
        if self.sock is not None:
            self.sock.send_multipart(data)
            data = self.sock.recv_multipart()
            value = pickle.loads(data[0])
            action = pickle.loads(data[1])
        else:
            with torch.no_grad():
                value, logp = self.model(torch.FloatTensor(state))
                value = value.item()
                action = logp.exp().multinomial(num_samples=1).item()
        product_strategy = ProductStrategy.to_type_id(action // len(NukeStrategy))
        nuke_strategy = NukeStrategy(action % len(NukeStrategy))
        return product_strategy, nuke_strategy


    def on_end(self, game_result):
        if self.sock is not None:
            score = 1. if game_result is Result.Victory else -1.
            """
            if game_result is Result.Victory:
                score = 0.5
            else: score = -0.5
            if self.bigDamage > 0:  #한번에 큰 대미지 넣은 횟수만큼
                score += self.bigDamage * 0.05
            print("리워드: ", score)"""
            self.sock.send_multipart((
                CommandType.SCORE, 
                pickle.dumps(self.game_id),
                pickle.dumps(score),
            ))
            self.sock.recv_multipart()



    '''def on_end(self, game_result):
        for tag in self.combatArray:
            print("111전투 : ", tag)
        print("-----")
        for tag in self.reconArray:
            print("2222222정찰 : ", tag)
        print("-----")
        for tag in self.nukeArray:
            print("핵 : ", tag)
        print("-----")'''