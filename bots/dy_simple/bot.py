
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time

import numpy as np

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    개별 전투 유닛이 적사령부에 바로 공격하는 대신, 15이 모일 때까지 대기
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.target_unit_counts = {
            UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
            UnitTypeId.MARINE: 25,
            UnitTypeId.MARAUDER: 15,
            UnitTypeId.REAPER: 0,
            UnitTypeId.GHOST: 0,
            UnitTypeId.HELLION: 0,
            UnitTypeId.SIEGETANK: 0,
            UnitTypeId.THOR: 0,
            UnitTypeId.MEDIVAC: 3,
            UnitTypeId.VIKINGFIGHTER: 0,
            UnitTypeId.BANSHEE: 0,
            UnitTypeId.RAVEN: 0,
            UnitTypeId.BATTLECRUISER: 0,
        }
        self.evoked = dict()

    async def on_step(self, iteration: int):
        """

        """
        # print(self.state.score.total_damage_dealt_life)
        actions = list() # 이번 step에 실행할 액션 목록

        ccs = self.units(UnitTypeId.COMMANDCENTER).idle  # 전체 유닛에서 사령부 검색
        combat_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        wounded_units = self.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치

        # 부족한 유닛 숫자 계산
        unit_counts = dict()
        for unit in self.units:
            unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
        
        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        #
        # 사령부 명령 생성
        #
        if ccs.exists:  # 사령부가 하나이상 존재할 경우
            cc = ccs.first  # 첫번째 사령부 선택
            next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련
            if self.can_afford(next_unit) and self.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(cc.train(next_unit))
                self.evoked[(cc.tag, 'train')] = self.time

        #
        # 유닛 명령 생성
        #
        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                #enemy_unit = self.known_enemy_units.health_least() #가장 체력이 적은 유닛???
                #print(enemy_unit.health)
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                target = enemy_cc
            else:
                target = enemy_unit

            """
            for unit in self.units.not_structure:
                near_unit = self.start_location
                if self.known_own_units.exists:
                    near_unit = self.known_own_units.closest_to(unit)  # 가장 가까운 아군 유닛
                    """
            """ 
            if near_unit.is_attacking: # 가까운 공격중인 아군 유닛이 있다면 아군 유닛의 타깃을 목표로 설정
                    target=near_unit.order_target
            else:
                # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
                if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                    target = enemy_cc
                else:
                    target = enemy_unit
            """

            if unit.type_id is not UnitTypeId.MEDIVAC:
                #print(combat_units.amount)
                if combat_units.amount > 15:
                    # 전투가능한 유닛 수가 15을 넘으면 적 본진으로 공격
                    actions.append(unit.attack(target))
                    use_stimpack = True
                else:
                    # 적 사령부 방향에 유닛 집결
                    target = self.start_location + 0.25 * (enemy_cc.position - self.start_location)
                    actions.append(unit.attack(target))
                    use_stimpack = False

                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if use_stimpack and unit.distance_to(target) < 15:
                        # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time

            if unit.type_id is UnitTypeId.MEDIVAC:
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                    #print(wounded_unit.name, "을 회복중")
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    actions.append(unit.move(combat_units.center))

                    ##아군 유닛이 공격중이 아니면 유닛 들이기(탑승)
                    ##내린지 1초가 지났고 근처 유닛들이 스팀팩 사용중이면(돌격준비가 되면)내리기
                    '''가장 가까운 유닛이 공격중인가를 기준으로 행동해서
                    탑승했던 유닛이 내리면 해당 유닛이 기준으로 변경됨
                    여기서 계속 태웠다가 내렸다가를 반복하는 것 같음'''
                    
                    '''명령을 내리고 실행까지 await로 기다리는 방식도 고려해봐야
                    if(유닛 몇개 이상이면):  
                        for i in self.units(combat_units).idle:  
                            await self.do(i.attack(target)) 꼴'''       
                    combat_unit = combat_units.closest_to(unit)
                    if self.time - self.evoked.get((unit.tag, AbilityId.LOAD_MEDIVAC), 0) > 1.0:
                        if not combat_unit.is_attacking:
                            actions.append(unit(AbilityId.LOAD_MEDIVAC, combat_unit))
                            self.evoked[(unit.tag, AbilityId.LOAD_MEDIVAC)] = self.time
                            print(combat_unit.name," (공격중:",combat_unit.is_attacking,") 탑승")
                    elif self.time - self.evoked.get((unit.tag, AbilityId.UNLOADALLAT_MEDIVAC), 0) > 1.0:
                        if combat_unit.has_buff(BuffId.STIMPACK):
                            actions.append(unit(AbilityId.UNLOADALLAT_MEDIVAC, unit.position))
                            self.evoked[(unit.tag, AbilityId.UNLOADALLAT_MEDIVAC)] = self.time
                            print("내림, 위치: ", unit.position)
                            print("남은자리:",unit.cargo_left,", 탑승중인 유닛 있는가?:",unit.has_cargo)



        await self.do_actions(actions)
        for a in actions:
            if a.ability == "LOAD_MEDIVAC":
                print("태우기: ", actions) ##출력안됨
        for a in actions:
            if a.ability == "UNLOADALLAT_MEDIVAC":
                print("내리기: ", actions) ##출력안됨

