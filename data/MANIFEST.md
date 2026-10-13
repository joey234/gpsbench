# GPSBench data manifest (v1.0)

Generated: 2026-09-21

- Main tasks: **17**
- Main rows on disk: **58,614**
- Main test rows (reported tables): **17,342**
- Diagnostic (Missing Data): **3,400**

Missing Data is not an 18th main task. Name Disambiguation test is 1,020 rows (470 unique questions). Coordinate Transformation is 2,039/339/1,022.

Source dump: `data/raw/cities15000.txt` has 32,709 rows and 244 country codes. Unsplit task JSONs live beside `splits/`; Name Disambiguation unsplit is the 4,214-row concat of its splits. `*.backup` files under `data/` are not part of the release.

| Task | Role | Train | Dev | Test | Total | Missing |
|---|---|---:|---:|---:|---:|---|
| task1_format_conversion | main | 2040 | 340 | 1020 | 3400 | — |
| task3_coordinate_system_transformation | main | 2039 | 339 | 1022 | 3400 | — |
| task4_distance_calculation | main | 2040 | 340 | 1020 | 3400 | — |
| task5_bearing_computation | main | 2040 | 340 | 1020 | 3400 | — |
| task6_interpolation | main | 2040 | 340 | 1020 | 3400 | — |
| task7_area_perimeter | main | 2040 | 340 | 1020 | 3400 | — |
| task8_bounding_box | main | 2040 | 340 | 1020 | 3400 | — |
| task9_route_geometry | main | 2040 | 340 | 1020 | 3400 | — |
| task10_relative_position | main | 2040 | 340 | 1020 | 3400 | — |
| task1_place_association | main | 2040 | 340 | 1020 | 3400 | — |
| task3_name_disambiguation | main | 2883 | 311 | 1020 | 4214 | — |
| task5_relative_position | main | 2040 | 340 | 1020 | 3400 | — |
| task6_proximity | main | 2040 | 340 | 1020 | 3400 | — |
| task7_route_analysis | main | 2040 | 340 | 1020 | 3400 | — |
| task8_boundary_analysis | main | 2040 | 340 | 1020 | 3400 | — |
| task9_spatial_patterns | main | 2040 | 340 | 1020 | 3400 | — |
| task13_terrain_classification | main | 2040 | 340 | 1020 | 3400 | — |
| task11_missing_data | diagnostic | 2040 | 340 | 1020 | 3400 | — |

## Checksums (SHA-256)

- `data/track_pure_gps/splits/task1_format_conversion_train.json` `ac853113f38790f83f5a4a5d690a799cfe550aaa26313f2fc817d08870a9870b`
- `data/track_pure_gps/splits/task1_format_conversion_dev.json` `1575fe110987bae78a36ed741384691b245864181e6971db8c3683c53a3454f9`
- `data/track_pure_gps/splits/task1_format_conversion_test.json` `236dafccf25bf2320838ae980aaa9c3dc02f6d92e690bd04f9c222bc7ead6cda`
- `data/track_pure_gps/splits/task1_format_conversion_split_info.json` `3bf88e2746050ed21f316da25a8e2b85418b6ae9beb4c0b9e68f31d2d55aefec`
- `data/track_pure_gps/splits/task3_coordinate_system_transformation_train.json` `d53c9fccfa5c31ce4d0517c3a5b95de602e1dc255fac4568a106c916a1dbc4e8`
- `data/track_pure_gps/splits/task3_coordinate_system_transformation_dev.json` `e27171e9a41535248fd3b5fa42c1ea5072e6f77e59bc78b2995c36186d994244`
- `data/track_pure_gps/splits/task3_coordinate_system_transformation_test.json` `5e1b12230be1e482a26a794be22660cfa92c776d5c7c4f5c36b944abc619a840`
- `data/track_pure_gps/splits/task3_coordinate_system_transformation_split_info.json` `83dcd615a80a3009566637a88e2e28c8584fd9779fc33bbc70b551bd3894213d`
- `data/track_pure_gps/splits/task4_distance_calculation_train.json` `c5214888d0cec8dc65cf00c026cf8743e0c38a7afec9ef7deb0c443977fa1a69`
- `data/track_pure_gps/splits/task4_distance_calculation_dev.json` `8e10434a901cab7c34b1e2ed98ac07017a0ded50cba17eeec8659e18cf967e0e`
- `data/track_pure_gps/splits/task4_distance_calculation_test.json` `509d62d25ff3dc6889d84421236eb6e59b4218ba1bdbc56a0257b565696be3c3`
- `data/track_pure_gps/splits/task4_distance_calculation_split_info.json` `180d0c36dd5c18c8f97b510887be5c0d025d521103e5f39823c2e174f378ad87`
- `data/track_pure_gps/splits/task5_bearing_computation_train.json` `b90e50386e7a11947ef3e7a1ced4c26c1c8d360fc1689912236022ef331f001b`
- `data/track_pure_gps/splits/task5_bearing_computation_dev.json` `61848c0943c3b5cbf798b4e73a9018dbd91a36059e875e6037fc611a8a8fcb4c`
- `data/track_pure_gps/splits/task5_bearing_computation_test.json` `5742377d855a45399fa41b22ed2ccf12641e326b4f21b29e285f73e63dbad107`
- `data/track_pure_gps/splits/task5_bearing_computation_split_info.json` `5f2bb3c46bd06710ce2dc698e4174b36114b49ecf75517ffd1d2bfc169fa39d3`
- `data/track_pure_gps/splits/task6_interpolation_train.json` `ba8d08ec93749d34b708b06a7ed66270f50048f58c5f54cc5672fccdf8399de1`
- `data/track_pure_gps/splits/task6_interpolation_dev.json` `7f8b66bc493b26ed1008fd525c56e2d3a17c3a7de3c05785e8add4bc6acce4b8`
- `data/track_pure_gps/splits/task6_interpolation_test.json` `7b574fc78fb71fb91cb8c713d62d9215554ded49abbd19d317720a29dfef803a`
- `data/track_pure_gps/splits/task6_interpolation_split_info.json` `1e2272050919c8ec3a8d14816f2f93e2a386d568332f8a2ba15c9761223d862c`
- `data/track_pure_gps/splits/task7_area_perimeter_train.json` `992c4fdf9739f12e7a10167ebb7a318418e8699fc228709e5f3062885c513fc3`
- `data/track_pure_gps/splits/task7_area_perimeter_dev.json` `ba586a7623d78ec5c202948bf3a4afc53f8161a6c5dde2551c8396933edd1eae`
- `data/track_pure_gps/splits/task7_area_perimeter_test.json` `ff90be14f6699474727b1aec38dd380637f40e7f6b597a6e1a4a8c02dfcaa435`
- `data/track_pure_gps/splits/task7_area_perimeter_split_info.json` `cfac02f05f4a55a3925944acdca092b65d20ea1e48c742a610ee4a969aa63b36`
- `data/track_pure_gps/splits/task8_bounding_box_train.json` `73f835e988216b9db9a44f1b805341979c28dae813de21e76c284622cb75cd34`
- `data/track_pure_gps/splits/task8_bounding_box_dev.json` `83c7633cab52d7a03babc2b94b55d4f9b60ca90504f936e78f85b46185abfce9`
- `data/track_pure_gps/splits/task8_bounding_box_test.json` `6dd22de48b3fa551f674d68d865fddf8321f36386f04f8e21240d2d74e50fff0`
- `data/track_pure_gps/splits/task8_bounding_box_split_info.json` `1162b3f45d0922718b730b5246cc31aeda51dc29abbab2f8a3a5a255b9f413d5`
- `data/track_pure_gps/splits/task9_route_geometry_train.json` `9c3a41e99c504ebb16fa6d80ec3d6cdac4fa977230b42957ef3af128cd0c474b`
- `data/track_pure_gps/splits/task9_route_geometry_dev.json` `1af5c9688819ca353e77266a5413012de62f7afeabfbd40dfe731d068a70a2a3`
- `data/track_pure_gps/splits/task9_route_geometry_test.json` `40e5bfb73e709e58889174b91c04732ff17e922368e230a99f1bd13268936653`
- `data/track_pure_gps/splits/task9_route_geometry_split_info.json` `cb1f36f3cd4ac01e4c297cb74594f720e1dbeb71b9784dd597a93201fb6639db`
- `data/track_pure_gps/splits/task10_relative_position_train.json` `57bb9e59dace36066aae8631e4dc8e9672401a110351b43382f810348da5b013`
- `data/track_pure_gps/splits/task10_relative_position_dev.json` `eb2f6f7023d7c14ef3b2a70faf523848c21907c884dd57e19196fec11204d20e`
- `data/track_pure_gps/splits/task10_relative_position_test.json` `a9e8ad6558c5a96ff502c0347dec2fa0fd259860103bece77b69f580035f0e9a`
- `data/track_pure_gps/splits/task10_relative_position_split_info.json` `5dbc5edd8a19f648b5550eac18a11b2671e99078ac5617e9bb6ccf1107c7129c`
- `data/track_applied/splits/task1_place_association_train.json` `a451735cf0d7208098860399aed522c9297c2f10a050ebca813bdab66b32ee9f`
- `data/track_applied/splits/task1_place_association_dev.json` `ea3925cc82dd3799e4891eb75e09381c7c96410fa5fde0fe3ce18cf0b73d0030`
- `data/track_applied/splits/task1_place_association_test.json` `2ac7052edfcf3d0a99316cb2250a8e6431abc4426c11a3efc4b8f9c7a63cb7a0`
- `data/track_applied/splits/task1_place_association_split_info.json` `771cdc457c1682efc1df495910dcaf72cf3365a687b7ca13c94c578f24963868`
- `data/track_applied/splits/task3_name_disambiguation_train.json` `f855248981b8e9ddfae65d71e44406647dfc86a17938d6958facadad39ecd17c`
- `data/track_applied/splits/task3_name_disambiguation_dev.json` `0a923cdcac26d8497bd8bb5f4e77bcc71041b36d64db8b8c9f9940095c17c3ad`
- `data/track_applied/splits/task3_name_disambiguation_test.json` `4e167f80c5c53bbd5db8e3c34ea8b103bca4630404fac10f1034fdb7525d65f0`
- `data/track_applied/splits/task3_name_disambiguation_split_info.json` `a488b6c7c4d0e4fbb11888aec0673d02db9bf56459b5c15017d84628f75b9b96`
- `data/track_applied/splits/task5_relative_position_train.json` `e808e24e4c1e7d27368f09ca8330f8d9af3324b969c3487d5af36a21dad5dc17`
- `data/track_applied/splits/task5_relative_position_dev.json` `20df62af7fa3e98fb54d9b3c338ec59265b6b161cf2b678a6c17a22e92151af1`
- `data/track_applied/splits/task5_relative_position_test.json` `470329bc182a1d55a448a8fd5a644e70808f9f3b7b65fcaadc2c3d59757ca9ac`
- `data/track_applied/splits/task5_relative_position_split_info.json` `da1681e6be09e37b66e846787cec0a81e40d94df7348b2f3ecd570b5319d3ed9`
- `data/track_applied/splits/task6_proximity_train.json` `31536e38c9abb3ff346982822a43fd8eca83b70a6ffbabb7b66e71948f83f7fb`
- `data/track_applied/splits/task6_proximity_dev.json` `edd628c8230ff7fc38a6ad6976e86382ca3515b87afdb97cdf4d52cf74846015`
- `data/track_applied/splits/task6_proximity_test.json` `0d3c8fe78c65b95caac30ee8f19344ce6ef480c7d959d109e22d2def8a05384e`
- `data/track_applied/splits/task6_proximity_split_info.json` `0fb6fd5e1d7ef2b32010bac1ff6358e35114f9f58610b441b9c2b87cb8fb310c`
- `data/track_applied/splits/task7_route_analysis_train.json` `5a6f37f45815bcf7a29eb39b305d2f0f1519fe7e736c9fb41440af4923cbe1c9`
- `data/track_applied/splits/task7_route_analysis_dev.json` `f58a5990f701bbdd3195f609187ad4b1e5f1250599a75b9e969fad57653f6de2`
- `data/track_applied/splits/task7_route_analysis_test.json` `dbe61423376aa7f8684d88a13b5b2864c85efc5e9f4188ee4968603a6c8b8ce2`
- `data/track_applied/splits/task7_route_analysis_split_info.json` `048b7e5156304fd99748d524fe28e2e978f975112e297f14fd294ee0caa39667`
- `data/track_applied/splits/task8_boundary_analysis_train.json` `6c508d216376f11d24bf6de1be69ccbde46e77002ba371c9f2d7cfb4254e0268`
- `data/track_applied/splits/task8_boundary_analysis_dev.json` `e6b70c6c62ab4ded9af63c392a8489dab4068e6a5651cb0c2b165990af492c54`
- `data/track_applied/splits/task8_boundary_analysis_test.json` `5dd31eb444d0a6830e4df6dc3ffd017dfdc58bac8a7c37ed07802c29c3c3765b`
- `data/track_applied/splits/task8_boundary_analysis_split_info.json` `2f36573237b3d30135cc34157e366400df618d5881ee8425f74669c4efe88b81`
- `data/track_applied/splits/task9_spatial_patterns_train.json` `98e509881ae9bc87c051d5ab48f8482595339759f0c2bb41fe48d78c39f3e1cc`
- `data/track_applied/splits/task9_spatial_patterns_dev.json` `356d2a10862ee47e7e3c4f36c6aca083dab026c7c0aebff4caabb8ec14864fa2`
- `data/track_applied/splits/task9_spatial_patterns_test.json` `0e481a58da53cedffde1b2658b34dc6c17c1afc54a8f3a188dbf2dfd6408ec59`
- `data/track_applied/splits/task9_spatial_patterns_split_info.json` `c2541b046f2063e6db38faa51438863c1f7360f321a990a806a5b188b1e721bc`
- `data/track_applied/splits/task13_terrain_classification_train.json` `eebc2e05a495b7427de6bea004c3da3183d7810a947f98de3911593645c84eba`
- `data/track_applied/splits/task13_terrain_classification_dev.json` `ab7871e4176d846bdc7027af0a1fd3e56e9bb5941c7ce7b347387e20ab22b551`
- `data/track_applied/splits/task13_terrain_classification_test.json` `42970b262300602d7d8d4d22e8110ba7e1eeaaa9e2f797d30df02696081bda3e`
- `data/track_applied/splits/task13_terrain_classification_split_info.json` `d740dd843a90bd7d2432ab0e98afcfbce8d458e9a38b8ebe602e71ee67ee5de9`
- `data/track_applied/splits/task11_missing_data_train.json` `2a34097731be76c16104d725d0e88fca94667824e7dc3ab6e63e509a2c13166d`
- `data/track_applied/splits/task11_missing_data_dev.json` `fdb7ed5a9d482c6a40a885bb31f0295a6cfd9a9e26963de8e792669be46d4a0a`
- `data/track_applied/splits/task11_missing_data_test.json` `6c0279e932d9f9091f3ce22544f76b874977193e598e458d1119d88fe1a89d10`
- `data/track_applied/splits/task11_missing_data_split_info.json` `6b23b007400adaa60fe70dae5206ccfb0315d886bef903f986b03ed95958f0b4`
