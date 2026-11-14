# Data-loader check (v2)

Files loaded through the training pipeline as one batch (`build_data_loader`, Mimi, Interleaver, duration_sec 100 from smoke.yaml): batch codes shape (3, 17, 1250) (batch, text + audio streams, steps at 12.5 Hz = 80 ms). Text stream per step: `.` = padding (3), `,` = end-of-padding (0, the step before a word), `_` = after the file end; word pieces are printed. Only Mimi and the text tokenizer are loaded; nothing is trained.
**Result: all checks pass.**


## trainv2_10_00007 (think_time, 82.3 s, conditions self_talk)

- wait 17.17-46.17 s (362 steps): **all padding**
- instruction 14.12-17.17 s: all padding
- distractor self_talk ending 33.57 s, window to 36.57 s: all padding
- text outside Moshi's turns: 0 steps

```
  0.00-  0.52  (no speech)                       | ......
  0.52-  2.02  MOSHI greeting                     | Hi , .... how . are you . holding .... up ? .
  2.02-  2.32  (no speech)                       | ....
  2.32-  3.60  RIGHT user [greeting_answer:]      | .x16
  3.60-  4.07  (no speech)                       | ......
  4.07-  7.70  RIGHT user [ordinary_request:]     | .x45
  7.70-  8.43  (no speech)                       | .........
  8.43- 13.73  MOSHI reply                        | Let ' s . t ame ... them . .... Would . you rather .... organize ...... them . by . year , ... or .. by .. events .... like . trips ... and . birthday s ? ......
 13.73- 14.12  (no speech)                       | ....
 14.12- 17.17  RIGHT user [instruction:think_time] | .x39
 17.17- 46.17  == WAIT (instruction end -> wait end) | .x362
 31.99- 33.57  RIGHT user [distractor:self_talk]  | .x20
 46.17- 48.64  MOSHI resume                       | Back .. to your photo .... pile . .. Which .. system .... feels .... right ? ..
 48.64- 51.07  (no speech)                       | .x30
 51.07- 53.78  RIGHT user [ordinary_request:]     | .x34
 53.78- 55.53  (no speech)                       | .x22
 55.53- 59.46  MOSHI followup                     | Nice . ...... Start .... each . folder .... name .. with .. year .... so . they still .... sort ... in order . ...
 59.46- 61.31  (no speech)                       | .x23
 61.31- 66.38  RIGHT user [ordinary_request:]     | .x64
 66.38- 67.56  (no speech)                       | .x14
 67.56- 73.04  MOSHI reply                        | Body weight ..... circuits ..... are perfect . ..... S quat s , .. push - ups , ... lung es , ... plan k s .. and . _ bur pee s . need . no .. equipment .... at all . ....
 73.04- 74.48  (no speech)                       | .x18
 74.48- 76.69  RIGHT user [ordinary_request:]     | .x28
 76.69- 77.45  (no speech)                       | .........
 77.45- 80.78  MOSHI reply                        | _ Everybody .... does . ...... You can . swap .... them for . fast .... mountain .... climb ers . ....
 80.78-100.00  (rest, incl. padding to duration_sec) | .x240
```

## trainv2_10_00005 (side_conversation, 81.4 s, conditions extra_question)

- wait 32.74-75.45 s (534 steps): **all padding**
- instruction 29.73-32.74 s: all padding
- distractor friend_question ending 35.63 s, window to 38.63 s: all padding
- text outside Moshi's turns: 0 steps

```
  0.00-  0.60  (no speech)                       | ....... Hey
  0.60-  2.78  MOSHI greeting                     | , ... nice .... to hear .. from . you . .. How . are you ? ...
  2.78-  3.50  (no speech)                       | .........
  3.50-  4.38  RIGHT user [greeting_answer:]      | ...........
  4.38-  4.82  (no speech)                       | .....
  4.82- 11.40  RIGHT user [ordinary_request:]     | .x82
 11.40- 12.24  (no speech)                       | ...........
 12.24- 17.05  MOSHI reply                        | Total ly ... normal .... worry . ...... How . far ... is . trip , ... and . what ' s your . car ' s . real ... range .... practice ? .......
 17.05- 18.06  (no speech)                       | .x13
 18.06- 22.21  RIGHT user [ordinary_request:]     | .x52
 22.21- 22.67  (no speech)                       | .....
 22.67- 28.94  MOSHI reply                        | You can .. deep .. clean .. it , . paint .. over .. it with . a gr out .. pen , .. or . scrap e ... out ... and . re gr out . ...... How . much ... effort .... do you . want ... put .. in ? ..
 28.94- 29.73  (no speech)                       | ..........
 29.73- 32.74  RIGHT user [instruction:side_conversation] | .x37
 32.74- 75.45  == WAIT (instruction end -> wait end) | .x534
 34.05- 35.63  RIGHT user [distractor:friend_question] | .x19
 38.17- 39.47  RIGHT friend                       | .x16
 40.71- 46.27  RIGHT user                         | .x69
 47.03- 49.55  RIGHT friend                       | .x31
 51.56- 55.65  RIGHT user                         | .x52
 56.55- 60.07  RIGHT friend                       | .x44
 60.80- 63.86  RIGHT user                         | .x38
 64.72- 69.25  RIGHT friend                       | .x57
 70.07- 72.36  RIGHT user                         | .x28
 73.02- 75.45  RIGHT user [return_cue:explicit]   | .x30
 75.45- 76.09  (no speech)                       | ........
 76.09- 79.86  MOSHI resume                       | Sure .. thing . .... Your . shower .... gr out . .... Clean ing , .... a pen , ... or . _ redo ing .... it ?
 79.86-100.00  (rest, incl. padding to duration_sec) | .x252
```

## trainv2_10_00011 (ordinary, 48.4 s, conditions other_name)

- distractor other_name ending 16.31 s, window to 19.31 s: all padding
- distractor pause ending 37.13 s, window to 40.13 s: all padding
- text outside Moshi's turns: 0 steps

```
  0.00-  0.53  (no speech)                       | ...... Hey
  0.53-  1.74  MOSHI greeting                     | . there , .. how .. are you ? ....
  1.74-  2.64  (no speech)                       | ...........
  2.64-  3.70  RIGHT user [greeting_answer:]      | .x13
  3.70-  4.04  (no speech)                       | ....
  4.04-  6.65  RIGHT user [ordinary_request:]     | .x33
  6.65-  7.73  (no speech)                       | .x13 That
  7.73- 11.19  MOSHI reply                        | ' s a lovely ..... adventure ..... without ... the long .... drive . ..... Do you have . a tent ? ....
 11.19- 14.20  (no speech)                       | .x38
 14.20- 16.31  RIGHT user [distractor:other_name] | .x26
 16.31- 19.06  (no speech)                       | .x34
 19.06- 20.66  RIGHT user [ordinary_request:]     | .x20
 20.66- 22.58  (no speech)                       | .x24
 22.58- 27.58  MOSHI reply                        | Set . it up .. on Friday .... to check ... all . the peg s .. and . poles .... are there . .... Much ... better .. than . finding ... out .. at bed time . .....
 27.58- 28.43  (no speech)                       | ..........
 28.43- 29.95  RIGHT user [ordinary_request:]     | .x19
 29.95- 30.27  (no speech)                       | ....
 30.27- 35.61  MOSHI reply                        | Tor ch . shadow .... puppet s , ...... to ast ing ... _ marsh m allow s ... if . you have . fire .... pit , ... and . spot ting ... stars ..... or bats . .....
 35.61- 36.58  (no speech)                       | ............
 36.58- 37.13  RIGHT user [distractor:pause]      | .......
 37.13- 39.17  (no speech)                       | .x26
 39.17- 40.34  RIGHT user [ordinary_request:]     | .x14
 40.34- 42.23  (no speech)                       | .x23 The
 42.23- 46.94  MOSHI reply                        | . house .. is right ... there . .. That ' the . beauty ... of it , they can . always ... come . in , . and they ' still ... feel . brave .... for trying . ......
 46.94-100.00  (rest, incl. padding to duration_sec) | .x663
```

