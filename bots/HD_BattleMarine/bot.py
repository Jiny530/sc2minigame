
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
        
        self.evoked = dict()


    async def on_step(self, iteration: int):
        """

        """
        # print(self.state.score.total_damage_dealt_life)
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.get_available_abilities(cc)
        combat_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        #다친 기계 유닛(탱크)
        wounded_units = self.units.filter(
            lambda u: u.is_mechanical and u.health_percentage < 0.5
        )  # 체력이 100% 이하인 유닛 검색

        enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치
        mule = self.units(UnitTypeId.MULE)
        mule_pos = cc.position.towards(enemy_cc.position, -5)


        if self.vespene > 200 :
            next_unit = UnitTypeId.BATTLECRUISER
        else:
            next_unit = UnitTypeId.MARINE

        
        #
        # 사령부 명령 생성
        #
        if self.can_afford(next_unit) and self.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
            # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
            actions.append(cc.train(next_unit))
            self.evoked[(cc.tag, 'train')] = self.time


        ##-----Mule 기계 유닛 힐-----
        if cc.health_percentage < 0.3: #cc피가 우선
            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_pos))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, cc))
        elif wounded_units.exists: #아픈 토르가 있는가?
            #공격중인 토르가 있는가?
            wounded_tank = wounded_units.filter(
                lambda u: u.is_attacking 
                and u.order_target in self.known_enemy_units.tags) 

            if wounded_tank.amount > 0:
                wounded_unit =wounded_tank.closest_to(cc) #'공격중인' 탱크있으면 걔 우선
            else: wounded_unit =wounded_units.closest_to(cc) #없으면 cc에 가까운 탱크 우선

            if mule.amount == 0:
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, wounded_unit.position))
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, wounded_unit))
        else: #택틱이 뮬인데 치료할거리가 없을경우(치료 끝난 후)
            if mule.amount > 0:
                mule_unit=mule.random
                if combat_units.exists:
                    actions.append(mule_unit.move(combat_units.center))
                else: actions.append(mule_unit.move(mule_pos))





        #
        # 유닛 명령 생성
        #
        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                target = enemy_cc
            else:
                target = enemy_unit

            if unit.type_id is not UnitTypeId.MEDIVAC:
                if combat_units.amount > 30:
                    # 전투가능한 유닛 수가 15를 넘으면 적 본진으로 공격
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
                        if unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time
 

        await self.do_actions(actions)

