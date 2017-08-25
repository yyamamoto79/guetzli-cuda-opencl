//
//  ViewController.m
//  guetzli_ios_metal
//
//  Created by 张聪 on 2017/8/10.
//  Copyright © 2017年 张聪. All rights reserved.
//

#import "ViewController.h"
#import "guetzli.h"
@interface ViewController ()
@property (nonatomic, retain) UITextView* tv;

@end

@implementation ViewController

-(instancetype)init{
    self = [super init];
    if (self) {
    }
    return self;
}

-(void)loadView{
    [super loadView];
//    UIImageView* imageV = [UIImageView new];
//    
//    imageV.image = [UIImage imageWithContentsOfFile:[NSString stringWithFormat:@"%@/%@",[[NSBundle mainBundle] resourcePath],@"1.png"]];
//    imageV.frame = CGRectMake(0, 100, imageV.image.size.width/2, imageV.image.size.height/2);
//    [self.view addSubview:imageV];
    UIButton* button = [UIButton buttonWithType:UIButtonTypeCustom];
    
    [button setTitle:@"开始测试" forState:UIControlStateNormal];
    [button setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
    
    button.frame = CGRectMake(50, 300 +20, 200, 80);
    button.backgroundColor = [UIColor yellowColor];
    [button addTarget:self action:@selector(onStartTest:) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:button];
    
    _tv = [UITextView new];
    _tv.frame = CGRectMake(50, 500, 300, 80);
    _tv.text = @"test";
    [self.view addSubview:_tv];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
}




-(void)onStartTest:(id)sender{
    NSLog(@"guetzli_ios_demo startTest");
    [self test1];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    NSLog(@"guetzli_ios_demo didReceiveMemoryWarning");
    // Dispose of any resources that can be recreated.
}

-(void)test1{
    
    NSString* input = [NSString stringWithFormat:@"%@/%@",[[NSBundle mainBundle] bundlePath],@"1.jpg"];//@"/Users/pinewang/Documents/guetzli_master/guetzli_ios/guetzli_ios_demo/resource/bees.png";
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths firstObject];
    NSString*output = [NSString stringWithFormat:@"%@/%@",documentsDirectory,@"1.jpg"];//@"/Users/pinewang/Documents/guetzli_master/guetzli_ios/guetzli_ios_demo/resource/testOut.png";
    NSFileManager* fm = [NSFileManager defaultManager];
    if ([fm fileExistsAtPath:output]) {
        [fm removeItemAtPath:output error:nil];
    }
    NSLog(@"guetzli_ios_demo test : input:%@,output:%@",input,output);
    NSTimeInterval before = [[NSDate date] timeIntervalSince1970];
    test([input UTF8String],[output UTF8String]);
    NSTimeInterval after = [[NSDate date] timeIntervalSince1970];
    NSTimeInterval time = after-before;
     NSLog(@"guetzli_ios_demo test Over,time:%f s",time);
    _tv.text = [NSString stringWithFormat:@"time:%f",time];
    
    NSString* path = output;
    

    
    UIImage* image = [UIImage imageWithContentsOfFile:path];
    

    
    UIImageView* imgView = [[UIImageView alloc] initWithImage:image];
    
    // 别忘了设置imageView的frame
    
    imgView.frame = CGRectMake(0.0f,0.0f,image.size.width,image.size.height);
     [self.view addSubview:imgView];
}
//    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
//        NSString* input = [NSString stringWithFormat:@"%@/%@",[[NSBundle mainBundle] bundlePath],@"1.jpg"];//@"/Users/pinewang/Documents/guetzli_master/guetzli_ios/guetzli_ios_demo/resource/bees.png";
//        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
//        NSString *documentsDirectory = [paths firstObject];
//        NSString*output = [NSString stringWithFormat:@"%@/%@",documentsDirectory,@"1.jpg"];//@"/Users/pinewang/Documents/guetzli_master/guetzli_ios/guetzli_ios_demo/resource/testOut.png";
//        NSFileManager* fm = [NSFileManager defaultManager];
//        if ([fm fileExistsAtPath:output]) {
//            [fm removeItemAtPath:output error:nil];
//        }
//        NSLog(@"guetzli_ios_demo test : input:%@,output:%@",input,output);
//        NSTimeInterval before = [[NSDate date] timeIntervalSince1970];
//        test([input UTF8String],[output UTF8String]);
//        NSTimeInterval after = [[NSDate date] timeIntervalSince1970];
//        NSTimeInterval time = after-before;
//        dispatch_async(dispatch_get_main_queue(), ^{
//            _tv.text = [NSString stringWithFormat:@"time:%f",time];
//        });
//        NSLog(@"guetzli_ios_demo test Over,time:%f s",time);
//    });
//}


@end
