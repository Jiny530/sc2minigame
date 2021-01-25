
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time

import numpy as np

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId
from enum import Enum
import random
import math


class CombatStrategy(Enum):
    OFFENSE = 0
    DEFENSE = 1

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


class CombatManager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):
        self.evoked = dict()  
        self.move_check = 0 #이동완료=1, 아니면 0 
        self.combat_pos = 1 #combat_units의 위치
        self.target_pos = self.bot.cc #이동 위치의 기준
        #0: defense / 1: offense - 고스트 사망 횟수에 따라 달라짐
        #self.mode = 0 #시작때는 defense

    def distance(self, pos1, pos2):
        result = math.sqrt( math.pow(pos1.x - pos2.x, 2) + math.pow(pos1.y - pos2.y, 2))
        return result

    def circle(self, target_pos):
        """
        target_pos를 중점으로해서 반지름이 3인 원으로 배치
        """
        x=[]
        y=[]

        for theta in range(0, 360):
            x.append(target_pos.x+r*math.cos(math.radians(theta)))
            y.append(target_pos.y+r*math.cos(math.radians(theta)))

    async def step(self):
        actions = list()

        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치
        cc_abilities = await self.bot.get_available_abilities(self.bot.cc)
        mule = self.bot.units(UnitTypeId.MULE)
        mule_pos = self.bot.cc.position.towards(enemy_cc.position, -5)
        #다친 기계 유닛(탱크)
        wounded_units = self.bot.units.filter(
            lambda u: u.is_mechanical and u.health_percentage < 0.5
        )  # 체력이 100% 이하인 유닛 검색

        #방어시 집결 위치
        #defense_position = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        def_pos1 = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        def_pos2 = self.bot.start_location + 0.5 * (enemy_cc.position - self.bot.start_location)
        def_pos3 = self.bot.start_location + 0.75 * (enemy_cc.position - self.bot.start_location)

        

        #wait_position = self.bot.start_location + 0.5 * (enemy_cc.position - self.bot.start_location)
        
        #얼마나 많은 유닛이 공격중인가?
        attacking_units = self.bot.combat_units.filter(
            lambda u: u.is_attacking 
            and u.order_target in self.bot.known_enemy_units.tags
        )  # 유닛을 상대로 공격중인 유닛 검색

        
        ##-----플래그 변경-----
        #3명 이상 공격중이면 fighting인걸로
        if attacking_units.amount > 5:
            self.bot.fighting = 1
        else: self.bot.fighting = 0

        ##-----Mule 기계 유닛 힐-----
        if self.bot.cc.health_percentage < 0.3: #cc피가 우선
            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_pos))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, self.bot.cc))
        elif wounded_units.exists: #아픈 토르가 있는가?
            #공격중인 토르가 있는가?
            wounded_tank = wounded_units.filter(
                lambda u: u.is_attacking 
                and u.order_target in self.bot.known_enemy_units.tags) 

            if wounded_tank.amount > 0:
                wounded_unit =wounded_tank.closest_to(self.bot.cc) #'공격중인' 탱크있으면 걔 우선
            else: wounded_unit =wounded_units.closest_to(self.bot.cc) #없으면 cc에 가까운 탱크 우선

            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(self.bot.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, wounded_unit.position))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, wounded_unit))
        else: #택틱이 뮬인데 치료할거리가 없을경우(치료 끝난 후)
            if mule.amount > 0:
                mule_unit=mule.random
                if self.bot.combat_units.exists:
                    actions.append(mule_unit.move(self.bot.combat_units.center))
                else: actions.append(mule_unit.move(mule_pos))

        #if(self.bot.combat_units.exists):
            #print("컴뱃센터: ", self.bot.combat_units.center.x)
            #print("combat_pos: ", self.combat_pos)
            #print("pos1: ", def_pos1)
            #print("pos2: ", def_pos2)
            #print("pos3: ", def_pos3)
            #print("move_check: ", self.move_check)
            #print("strategy: ", self.bot.combat_strategy)
        
        #
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        for unit in self.bot.units.not_structure:  # 건물이 아닌 유닛만 선택
            if unit in self.bot.combat_units:
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
                    #if len(self.bot.combatArray) > 20 and self.bot.units.of_type([UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK]).amount > 1:
                    if self.bot.combat_strategy == CombatStrategy.DEFENSE: #1=DEFENSE
                        self.target_pos = def_pos1
                        actions.append(unit.attack(self.target_pos))
                    elif self.bot.combat_strategy == CombatStrategy.OFFENSE: #0=OFFENSE         
                            if self.combat_pos == 1 and self.move_check == 0:
                                #print("오펜스됨")
                                self.target_pos = def_pos2
                                actions.append(unit.attack(self.target_pos))
                                if self.distance(self.bot.combat_units.center, def_pos2) < 3:
                                    #print("pos1도착")
                                    self.combat_pos = 2
                                    self.move_check = 1
                            elif self.combat_pos == 2 and self.move_check == 0:
                                self.target_pos = def_pos3
                                actions.append(unit.attack(self.target_pos))
                                if self.distance(self.bot.combat_units.center, def_pos3) < 3:
                                    #print("pos2도착")
                                    self.combat_pos = 3
                                    self.move_check = 1
                            elif self.combat_pos == 3 and self.move_check == 0:
                                actions.append(unit.attack(self.target_pos))

                    ##-----해병과 불곰-----
                    if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                        #OFFENSE
                        if self.bot.combat_strategy == CombatStrategy.OFFENSE and unit.distance_to(target) < 15:
                            # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                            if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                                # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                                if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                    # 1초 이전에 스팀팩을 사용한 적이 없음
                                    actions.append(unit(AbilityId.EFFECT_STIM))
                                    self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time
                    
                    ##-----탱크-----
                    if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                        if self.bot.combat_strategy == CombatStrategy.DEFENSE:
                            #print("디펜스임-탱크")
                            if self.combat_pos == 1 and self.move_check == 0:
                                #print("초기조건 들어감")
                                if unit.type_id == UnitTypeId.SIEGETANK:
                                    #print("탱크인거 알고 있음")
                                    order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                    actions.append(order)
                                elif unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                    self.move_check = 1
                        elif self.bot.combat_strategy == CombatStrategy.OFFENSE:
                            if self.move_check == 1: #이동완료
                                if unit.type_id == UnitTypeId.SIEGETANK:
                                    order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                    actions.append(order)
                                elif unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                    self.move_check = 0
                            elif self.move_check == 0: #이동해야 
                                if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                    order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                    actions.append(order)
                        
                        '''#상태에 따라 분류
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
                            #타겟설정: 해병과 불곰 우선
                            if self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).exists:
                                target = self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).closest_to(unit)
                        '''
        return actions

class AssignManager(object): #뜯어고쳐야함
    """
    유닛을 부대에 배치하는 매니저
    *reconArray: recon에 들어가는 해병만 있음
    *nukeArray: 고스트만 있음
    *combatArray: combat의 해병, 탱크, 밤까마귀 있음

    *combat_units: combat의 해병, 탱크만 있음(combatArray보다 작은 개념)
    *nuke_units: 고스트만 있음=nukeArray
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

        #이미 할당된 유닛의 태그 빼고
        units_tag = units_tag - self.bot.combatArray


        #------유닛 타입에 따라 array 배정(레이븐은 패스)-----
        for tag in units_tag:
            unit = self.bot.units.find_by_tag(tag)
            if unit.type_id in (UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK): #탱크(변신)는 컴뱃
                self.bot.combatArray.add(unit.tag)
            elif unit.type_id is UnitTypeId.MARINE: #마린도 컴뱃으로
                self.bot.combatArray.add(unit.tag)



class TrainManager(object):
    """
    CombatManager의 생산 유닛을 결정
    유령과 밤까마귀는 각 매니저에서 생산 결정
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai
    
    def next_unit(self):
        if self.bot.vespene > 200:
            next_unit = UnitTypeId.SIEGETANK
        else:
            next_unit = UnitTypeId.MARINE

        return next_unit


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    개별 전투 유닛이 적사령부에 바로 공격하는 대신, 15이 모일 때까지 대기
    """
    def __init__(self, step_interval=5.0, *args, **kwargs):
        super().__init__()
        self.step_interval = step_interval
        self.step_manager = StepManager(self)
        self.combat_manager = CombatManager(self)
        self.train_manager = TrainManager(self)
        self.assign_manager = AssignManager(self)

        self.combatArray = set()

        self.productIng = 0 

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.evoked = dict()

        self.combat_strategy = CombatStrategy.DEFENSE

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        self.step_manager.reset()
        self.combat_manager.reset()
        self.assign_manager.reset()

        self.assign_manager.assign()


    async def on_step(self, iteration: int):
        """

        """
        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록

        #부대별 units타입
        self.combat_units = self.units.filter(
            lambda unit: unit.tag in self.combatArray
            and unit.type_id in [UnitTypeId.MARINE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )

        if self.time - self.last_step_time >= self.step_interval:
            #택틱 변경  
            if self.combat_units.amount > 20:
                self.combat_strategy = CombatStrategy(0)
                self.combat_manager.move_check = 0
            #n = random.randint(0,1)
            #self.combat_strategy = CombatStrategy(n)
            #print("택틱: ", self.combat_strategy)
            #print("컴뱃어레이 길이: ", len(self.combatArray))

            self.last_step_time = self.time

    
        if self.productIng == 0: 
            actions += await self.train_action() #생산
            self.productIng = 1 #생산명령 들어갔다고 바꿔줌

        #생산 명령이 들어갔다면
        elif self.productIng == 1:
            #배치
            self.assign_manager.assign()
            self.productIng = 0

        actions += await self.combat_manager.step()   
        
        ## -----명령 실행-----
        await self.do_actions(actions)


    async def train_action(self):
        #
        # 사령부 명령 생성
        #
        actions = list()
        next_unit = self.train_manager.next_unit()
        
        if self.can_afford(next_unit):
            if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(self.cc.train(next_unit))
                self.evoked[(self.cc.tag, 'train')] = self.time
        return actions

