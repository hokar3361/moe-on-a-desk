#!/bin/bash
# mac125b engine console: ~/console/console.sh [start|stop|status]
D=$HOME/console
case "${1:-start}" in
  start)
    if pgrep -f "[c]onsole/console.py" > /dev/null; then echo "already running"; else
      nohup python3 $D/console.py > $D/console.log 2>&1 < /dev/null &
      sleep 2
    fi
    head -1 $D/console.log ;;
  stop)
    pkill -f "[c]onsole/console.py"; echo "console stopped (engines keep running; stop them from the page first)" ;;
  status)
    pgrep -fl "[c]onsole/console.py" || echo "not running"; head -1 $D/console.log ;;
esac
