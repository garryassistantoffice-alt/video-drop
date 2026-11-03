# video-drop
Finished videos, ready to download and post.

This is a shelf, not an archive (Alex, 30 October 2026: "I want everything that already been posted to be gone from
video drop repo after a week"): a week after a video is posted on YouTube, its files (the video, its subtitles and its
YouTube text) are taken off this shelf. Claude checks once a week. The source of every video (scripts, pictures,
subtitles) stays in the Garry Forge repository.

Links for Alex's iPad (3 November 2026): Safari shows a plain-git subtitles or text file (raw.githubusercontent.com)
as text, so YouTube Studio's "Upload file" cannot find it. A subtitles file Alex uploads himself also gets a copy kept
in Git LFS (a line in .gitattributes), and the link he gets is media.githubusercontent.com/media/<owner>/<repo>/<commit>/
<path>: it comes as a download, like the videos. Example: opera-stories-figaro/full/figaro-full-subtitles.srt.
