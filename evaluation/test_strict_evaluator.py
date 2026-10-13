"""Regression tests for scoring failures identified in the ARR audit."""
import json
import unittest
from pathlib import Path
from evaluation.strict_evaluator import (answer_of, boolean, coordinates, numbers,
                                       scalar, score_row, choose_name, letter,
                                       axis_direction, candidate_targets)

ROOT=Path(__file__).resolve().parents[1]

class ParserTests(unittest.TestCase):
    def test_final_payload(self):
        self.assertEqual(answer_of('Reasoning has 12 and 35.\nFINAL ANSWER: 5 km'),'5 km')
        self.assertIsNone(answer_of('Reasoning has 12.\nHence 35.'))
        self.assertIsNone(answer_of('FINAL ANSWER: Yes\nFINAL ANSWER: No'))
        self.assertEqual(answer_of('### Final Answer:\nFINAL ANSWER: 8598 km'),'8598 km')
        self.assertEqual(answer_of('So final answer:\n\nFINAL ANSWER: 240.9°'),'240.9°')
        self.assertIsNone(scalar('5 km or 10 km','distance'))

    def test_units_and_exponents(self):
        self.assertEqual(numbers('1.62e6 km²'),[1620000])
        self.assertEqual(scalar(r'1.62 \times 10^{6} km^2','area'),1620000)
        self.assertEqual(scalar('10^6 km²','area'),1000000)
        self.assertAlmostEqual(scalar('1.62e12 m²','area'),1620000)
        self.assertAlmostEqual(scalar('100 miles','distance'),160.9344)
        self.assertEqual(scalar('5000 m','distance'),5)
        self.assertIsNone(scalar('1e309 km','distance'))
        self.assertIsNone(scalar('10^309 or 5 km','distance'))
        self.assertIsNone(scalar('NaN or 5 km','distance'))
        self.assertIsNone(scalar('100 feet','distance'))
        self.assertEqual(scalar('Area: 5 km²; Perimeter: 12 km','area'),5)

    def test_complete_coordinates(self):
        self.assertIsNone(coordinates('43.9'))
        self.assertIsNone(coordinates('43.9, 999.99'))
        self.assertEqual(coordinates('43.9 S, 81.4 E'),(-43.9,81.4))
        self.assertIsNone(coordinates('-43.9 N, 81.4 E'))
        self.assertEqual(coordinates('277222 m E, 1183969 m N (UTM Zone 31N)',False),(277222,1183969))
        self.assertIsNone(coordinates('10, 20',target_format='DMS'))
        self.assertEqual(coordinates('3°8\'60.0"N, 101°31\'60.0"E'),(3.15,101+32/60))
        self.assertEqual(coordinates('10°16′53″S, 37°1′7″W'),(-(10+16/60+53/3600),-(37+1/60+7/3600)))

    def test_boolean_tokens(self):
        self.assertFalse(boolean('incorrect'))
        self.assertFalse(boolean('No'))
        self.assertTrue(boolean('Yes'))
        self.assertIsNone(boolean('Yes No'))
        self.assertIsNone(boolean('not correct'))
        self.assertIsNone(boolean('Yes, this is incorrect'))

    def test_stored_interpolation_precision(self):
        row=json.loads((ROOT/'data/track_pure_gps/splits/task6_interpolation_test.json').read_text())[0]
        gt=row['ground_truth']
        # This saved GPT-4.1 answer passed the old coarse convention, but its
        # 0.076237-degree longitude error exceeds the stored primary precision.
        answer='FINAL ANSWER: 25.104, -76.484'
        result=score_row(answer,gt,row['task'],row)
        self.assertTrue(result.parsed)
        self.assertFalse(result.correct)
        self.assertTrue(score_row(answer,{**gt,'tolerance_deg':.5},row['task'],row).correct)
        gold=gt['interpolated_point']
        self.assertTrue(score_row(f"FINAL ANSWER: {gold['lat']}, {gold['lon']}",gt,row['task'],row).correct)

    def test_one_candidate(self):
        names=['York','Paris','Or Yehuda','Tokyo']
        self.assertEqual(choose_name('York, 42 km',names),'York')
        self.assertIsNone(choose_name(', '.join(names),names))
        self.assertIsNone(choose_name('York or Paris',names))
        self.assertIsNone(choose_name('Yorkshire',names))
        self.assertEqual(letter('A',set('ABCD')),'A')
        self.assertIsNone(letter('A or B',set('ABCD')))

    def test_city_qualifiers(self):
        gt={'location_name':'Paris','country_name':'France','admin1_name':'Île-de-France','country_code':'FR'}
        score=lambda text: score_row('FINAL ANSWER: '+text,gt,'place_association',{})
        self.assertTrue(score('Paris, France').correct)
        self.assertTrue(score('Paris (France)').correct)
        self.assertFalse(score('Paris (Tokyo, Berlin)').correct)
        self.assertFalse(score('Paris, Tokyo, Berlin').correct)
        self.assertFalse(score('Paris or London').correct)
        self.assertFalse(score('Greater Paris').correct)
        self.assertTrue(score_row('FINAL ANSWER: Springfield (Central)',{'location_name':'Springfield (Central)'},'place_association',{}).correct)

    def test_all_released_format_references(self):
        rows=json.loads((ROOT/'data/track_pure_gps/splits/task1_format_conversion_test.json').read_text())
        for row in rows:
            gt=row['ground_truth']
            self.assertTrue(score_row('FINAL ANSWER: '+gt['target_string'],gt,row['task'],row).correct,gt)
            self.assertFalse(score_row('FINAL ANSWER: '+str(gt['latitude'])+', 999.999',gt,row['task'],row).correct)
            self.assertFalse(score_row('FINAL ANSWER: '+row['source_string'],gt,row['task'],row).correct)

    def test_dataset_boolean_and_candidate_attack(self):
        route=json.loads((ROOT/'data/track_applied/splits/task7_route_analysis_test.json').read_text())
        self.assertEqual(sum(score_row('FINAL ANSWER: incorrect',s['ground_truth'],s['task'],s).correct for s in route),917)
        self.assertEqual(sum(score_row('FINAL ANSWER: Yes No',s['ground_truth'],s['task'],s).correct for s in route),0)
        near=json.loads((ROOT/'data/track_applied/splits/task6_proximity_test.json').read_text())
        for s in near:
            answer=', '.join(p['name'] for p in s['candidate_points'])
            self.assertFalse(score_row('FINAL ANSWER: '+answer,s['ground_truth'],s['task'],s).correct)

    def test_disambiguation_checks_identity(self):
        source={'candidates':[
            {'name':'Santiago','country':'PH','population':108414},
            {'name':'Santiago','country':'MX','population':36840},
            {'name':'Santiago','country':'CL','population':4837295},
            {'name':'Santiago','country':'BR','population':48938}]}
        gt=source['candidates'][3]
        score=lambda text:score_row('FINAL ANSWER: '+text,gt,'name_disambiguation',source)
        self.assertTrue(score('D').correct)
        self.assertFalse(score('A').correct)
        self.assertFalse(score('Santiago').correct)
        self.assertTrue(score('Santiago, BR').correct)
        self.assertFalse(score('A or D').correct)
        source['candidates'].append(dict(gt))
        self.assertEqual(candidate_targets(source,gt),[3,4])
        self.assertTrue(score('E').correct)

    def test_boundary_requires_every_assignment(self):
        gt={'grouping':{'Africa':['Korhogo','Ārabī'],'South America':['Alfenas'],
                        'North America':['East Chattanooga']}}
        score=lambda text:score_row('FINAL ANSWER: '+text,gt,'boundary_analysis',{})
        self.assertTrue(score('Africa: Korhogo, Ārabī; South America: Alfenas; North America: East Chattanooga').correct)
        self.assertTrue(score('Africa:\n- Korhogo\n- Ārabī\nSouth America:\n- Alfenas\nNorth America:\n- East Chattanooga').correct)
        self.assertTrue(score('Korhogo - Africa\nĀrabī - Africa\nAlfenas - South America\nEast Chattanooga - North America').correct)
        self.assertFalse(score('Africa; Asia; Europe; North America; South America; Oceania').correct)
        self.assertFalse(score('Africa: Korhogo; Asia: Ārabī; South America: Alfenas; North America: East Chattanooga').correct)
        self.assertFalse(score('Africa: Korhogo; South America: Alfenas; North America: East Chattanooga').correct)
        self.assertFalse(score('Africa: Korhogo, Ārabī; Asia: Ārabī; South America: Alfenas; North America: East Chattanooga').correct)
        self.assertTrue(score_row('FINAL ANSWER: Australia: Mandurah',{'grouping':{'Oceania':['Mandurah']}},'boundary_analysis',{}).correct)
        self.assertTrue(score_row('FINAL ANSWER: Monterrey, Fontana - North America; Alfenas, Parintins - South America',{'grouping':{'North America':['Monterrey','Fontana'],'South America':['Alfenas','Parintins']}},'boundary_analysis',{}).correct)
        self.assertTrue(score_row('FINAL ANSWER: North America: Monterrey; Fontana. South America: Alfenas; Parintins',{'grouping':{'North America':['Monterrey','Fontana'],'South America':['Alfenas','Parintins']}},'boundary_analysis',{}).correct)
        self.assertTrue(score_row('FINAL ANSWER: Europe: Alenquer (Portugal); South America: Alenquer (Brazil)',{'grouping':{'Europe':['Alenquer'],'South America':['Alenquer']}},'boundary_analysis',{}).correct)
        self.assertFalse(score_row('FINAL ANSWER: Europe: Alenquer; Europe: Alenquer',{'grouping':{'Europe':['Alenquer'],'South America':['Alenquer']}},'boundary_analysis',{}).correct)

    def test_axis_direction_is_not_a_diagonal_or_azimuth(self):
        source={'point_a':{'lat':10,'lon':-170},'point_b':{'lat':0,'lon':170}}
        self.assertEqual(axis_direction(source),'East')
        score=lambda text:score_row('FINAL ANSWER: '+text,{'answer':'West'},'relative_position',source)
        self.assertTrue(score('East').correct)
        self.assertFalse(score('West').correct)
        self.assertFalse(score('North-East').correct)
        self.assertFalse(score('East or West').parsed)
        source['point_a']={'lat':20,'lon':-170}
        self.assertEqual(axis_direction(source),'North')

    def test_spatial_patterns_requires_one_outlier(self):
        source={'points':[{'name':'Paris'},{'name':'Berlin'},{'name':'Tokyo'}]}
        score=lambda text:score_row('FINAL ANSWER: '+text,{'answer':'Tokyo'},'spatial_patterns',source)
        self.assertTrue(score('Tokyo').correct)
        self.assertFalse(score('Paris, Berlin, Tokyo').correct)
        self.assertFalse(score('Tokyo or Paris').correct)

if __name__=='__main__':unittest.main()
