# TODO LIST
- [ ] Revamp RecordingFragment
      -> centralize stop recording mark (P2)
      -> DONE -- select video from gallery (P2)
      -> DONE -- show recorded video after stop on ConfirmationView, before processing the video (P0)
      -> implement real processing on processing button clicked (P0)
         -> [OPENCV, read video] rPPG to extract timeline of heartbeats
         -> extract video duration
         -> analyze heartbeats data to generate arrhythmia diagnosis
         -> [OPENCV, create video] video with heartbeat markers on each heartbeats
         -> [OPENCV, create image] image of heartbeat timeline
- [ ] Revamp DetailFragment
      -> DONE -- change video player to vertical position (9x16)
      -> DONE -- disable autoplay video on open detail fragment
      -> DONE -- zoom in image on tap (P2)
      -> adjust text and image position after video player adjustment

# BUGS LIST
- [x] Video recorded with front camera is mirrored upside-down (P0)
- [x] Redirection RecordFragment to DetailFragment is stacked, replace with replace instead 
- [ ] Stuck auto-increment record data for new data
- 