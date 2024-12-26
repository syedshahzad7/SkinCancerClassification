
(function($) {
	'use strict';
	
	jQuery(document).on('ready', function(){
	
		/*PRELOADER JS*/
		$(window).on('load', function() { 
			$('.status').fadeOut();
			$('.preloader').delay(350).fadeOut('slow'); 
		}); 
		/*END PRELOADER JS*/		
			
		/*START MENU JS*/		
		function windowScroll() {
			const navbar = document.getElementById("navbar");
			if (
				document.body.scrollTop >= 50 ||
				document.documentElement.scrollTop >= 300
			) {
				navbar.classList.add("nav-sticky");
			} else {
				navbar.classList.remove("nav-sticky");
			}
		}

		window.addEventListener('scroll', (ev) => {
			ev.preventDefault();
			windowScroll();
            
		})	  	
		/*END MENU JS*/	 
		
	}); 	 
    
})(jQuery);

//Countdow JS

$(document).ready(function() {

    $('.counter').each(function () {
        $(this).prop('Counter',0).animate({
            Counter: $(this).text()
        }, {
            duration: 4000,
            easing: 'swing',
            step: function (now) {
                $(this).text(Math.ceil(now));
            }
        });
    });

});

// bottom to up
$(window).on('scroll', function () { 
    var scrolled = $(window).scrollTop();
    if (scrolled > 400) $('.go-top').addClass('active');
    if (scrolled < 400) $('.go-top').removeClass('active');
});


$('.go-top').on('click', function () {
    $("html, body").animate({
        scrollTop: "0"
    }, 50);
});



//Mamuer beta slider

$('.owl-carousel').owlCarousel({
    loop:true,
    margin:16,
    responsiveClass:true,
    autoplay: false, 
    autoplaySpeed: 1500,
    dots: true,
    responsive:{
        0:{
            items:1,
            nav:true
        },
        350:{
            margin:6,
            items:1,
            nav:false
        },
        575:{
            margin:10,
            items:2,
            nav:false
        },
        1000:{
            items:3,
            nav:true,
            loop:true
        },
        1920:{
            items:4,
            nav:true,
            loop:true
        }
    }
}); 


$(".scroll").on("click", function(event){
    event.preventDefault();
    $('html,body').animate({
        scrollTop: $(this.hash).offset().top - 130});
});
