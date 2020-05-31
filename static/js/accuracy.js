var angle,
          valueUnits,
          markerAngle,
          lastAngle = 0,
          circle = $('#svg-chart_pie > circle');
      
      $.each(circle, function(){
      
          $(this).css({
              transform: 'rotate(' + lastAngle + 'deg)'
          });
      
          angle = (360 / 100) * $(this).attr('aria-valuenow');
          markerAngle = lastAngle + (angle / 2);
          lastAngle = lastAngle + angle;
      
          $(this).attr('data-marker-angle', Math.ceil(markerAngle));
      
          $(this).click(function(){
              $('[name="goldenbox-controls"]:eq(' + $(this).index() + ')').click();
              $('#goldenbox-close-control').attr('checked', false);
              $('#active-icon').attr('xlink:href', '#' + $(this).attr('data-slice-name'));
              $('#svg-chart_info_marker').attr('data-active', $(this).attr('data-slice-name'));
              console.log();
              $('#svg-chart_info_marker').css({
                  transform: 'rotate(' + $(this).attr('data-marker-angle') + 'deg)'
              });
          });
      });
      
      setTimeout(function()
      {
          circle.eq(0).click();
      }, 1500);