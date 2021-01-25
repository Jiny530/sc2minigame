
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time

import numpy as np

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
import random
import math



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
        self.move_check = 0 #전부 이동완료=1, 아니면 0 
        self.move_tank = 0 #이동 완료한 탱크
        self.combat_pos = 0 #combat_units의 위치
        self.target_pos = self.bot.cc #이동 위치의 기준
        self.position_list = list() #탱크 유닛들의 포지션 리스트

        self.i = 0 #x쪽 가중치
        self.j = 0 #y쪽 가중치
        self.n = 0


    def distance(self, pos1, pos2):
        """
        두 점 사이의 거리
        """
        result = math.sqrt( math.pow(pos1.x - pos2.x, 2) + math.pow(pos1.y - pos2.y, 2))
        return result

        
    def circle(self, target_pos):
        """
        target_pos를 중점으로해서 지그재그 배치
        """
        
        self.position_list.append(Point2((target_pos.x + self.i, target_pos.y + self.j)))
        if(self.n%2 == 1):
            self.i = self.i + 1 #x좌표는 홀수번째마다 증가
            self.j = self.j + self.n #y좌표는 홀수에 증가, 짝수에 감소
        else:
            self.j = self.j - self.n

            
    def moving(self, unit, pos, actions):
        t=0
        if len(self.position_list) == self.n : #리스트 갱신
            print(len(self.position_list),"@@@@@@@@@",self.n)
            self.circle(pos)
        for tank in self.bot.tankArray: #이동
            if unit.tag == tank and t<=self.n:
                actions.append(unit.move(self.position_list[t]))  
                #print("!!!!!!!!!!!!!!!!!")
            t+=1
        if self.distance(unit.position, self.position_list[self.n]) < 1: #지정위치도착
            if unit.type_id == UnitTypeId.SIEGETANK: 
                if self.move_check==0:
                    print(self.move_check,"!!!!!!!!!!",self.n,"@@@@@@",t)
                order = unit(AbilityId.SIEGEMODE_SIEGEMODE) #변신함
                actions.append(order)
        self.n = self.n + 1 
        
    #region Description
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
    #endregion
        
         #방어시 집결 위치
        #defense_position = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
        def_pos1 = self.bot.start_location + 0.20 * (enemy_cc.position - self.bot.start_location)
        def_pos2 = self.bot.start_location + 0.45 * (enemy_cc.position - self.bot.start_location)
        def_pos3 = self.bot.start_location + 0.7 * (enemy_cc.position - self.bot.start_location)
        
        _pos1 = self.bot.start_location + 0.30 * (enemy_cc.position - self.bot.start_location)
        _pos2 = self.bot.start_location + 0.55 * (enemy_cc.position - self.bot.start_location)
        _pos3 = self.bot.start_location + 0.8 * (enemy_cc.position - self.bot.start_location)

        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        self.n = 0
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

#region marine
                ##-----전투 유닛 전체-----
                if unit.type_id is UnitTypeId.MARINE:
                    if self.move_check == 0: #초기상태+대기중
                        #print("해병: 초기상태+대기중")
                        self.target_pos = _pos1
                        actions.append(unit.attack(self.target_pos))
                    elif self.move_check == 1: 
                        self.target_pos = _pos2
                        actions.append(unit.attack(self.target_pos))
                    elif self.move_check == 2: #초기상태+대기중
                        #print("해병: 초기상태+대기중")
                        self.target_pos = _pos3
                        actions.append(unit.attack(self.target_pos))
                    if self.distance(self.bot.combat_units.center, _pos3) < 2: #위치3도착
                        actions.append(unit.attack(target))

                    ##-----스킬-----
                    if self.combat_pos == 1 and unit.distance_to(target) < 15:
                            # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time

                    ##-----타겟-----
                    #타겟 우선순위 설정
                    if self.bot.known_enemy_units.exists:
                        flying_target = self.bot.known_enemy_units.filter(
                            lambda unit:  unit.is_flying
                        )
                        if flying_target.exists:
                            target = flying_target.closest_to(unit)

#endregion
                    ##-----탱크-----
                if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):

                    if self.move_check == 0: #초기상태+대기중
                        self.moving(unit, def_pos1, actions)
                        '''t=0
                        if len(self.position_list) == self.n : #리스트 갱신
                            self.circle(pos)
                        for tank in self.bot.tankArray: #이동
                            if unit.tag == tank and t<=self.n:
                                actions.append(unit.move(self.position_list[t]))  
                            t+=1
                        if self.distance(unit.position, self.position_list[self.n]) < 1: #지정위치도착
                            if unit.type_id == UnitTypeId.SIEGETANK: 
                                order = unit(AbilityId.SIEGEMODE_SIEGEMODE) #변신함
                                actions.append(order)
                        self.n = self.n + 1 '''
                        
                    elif self.move_check == 1: 
                        self.moving(unit, def_pos2, actions)
                    elif self.move_check == 2: 
                        self.moving(unit, def_pos3, actions)


        if self.bot.combat_units.amount > 40 and self.move_check == 0:
            #print("20넘음")
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            self.i = 0
            self.j = 0
            self.move_check = 1
        elif self.bot.combat_units.amount > 80 and self.move_check == 1:
            #print("40넘음")
            for unit in self.bot.combat_units(UnitTypeId.SIEGETANKSIEGED):
                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                actions.append(order)
            self.position_list = list()
            self.i = 0
            self.j = 0
            self.move_check = 2
                          
                        
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
                self.bot.tankArray.append(unit.tag)
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
        self.tankArray = list()

        self.productIng = 0 

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.evoked = dict()


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

        

        if self.time - self.last_step_time >= self.step_interval:
            #택틱 변경  
            """
            if self.combat_units.amount > 20 and self.combat_manager.combat_pos == 1:
                print("20넘음")
                self.combat_manager.move_check = 0
            elif self.combat_units.amount > 40 and self.combat_manager.combat_pos == 2:
                print("40넘음")
                self.combat_manager.move_check = 0
            elif self.combat_units.amount > 60 and self.combat_manager.combat_pos == 3:
                print("60넘음")
                self.combat_manager.move_check = 0"""
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

        #부대별 units타입
        self.combat_units = self.units.filter(
            lambda unit: unit.tag in self.combatArray
            and unit.type_id in [UnitTypeId.MARINE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )
        self.tank_units = self.combat_units.filter(
            lambda unit:  unit.type_id in [UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]
        )
        self.combat_marine_units = self.combat_units.filter(
            lambda unit:  unit.type_id is UnitTypeId.MARINE
        )

        actions += await self.combat_manager.step()   

        #print(self.tank_units.amount, "0****************", len(self.tankArray))


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

