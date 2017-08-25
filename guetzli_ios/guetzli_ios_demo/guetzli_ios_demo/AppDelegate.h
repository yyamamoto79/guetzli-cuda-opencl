//
//  AppDelegate.h
//  guetzli_ios_demo
//
//  Created by pine wang on 17/6/28.
//  Copyright © 2017年 com.tencent. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreData/CoreData.h>

@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@property (strong, nonatomic) UIViewController *rootController;

@property (readonly, strong) NSPersistentContainer *persistentContainer;

- (void)saveContext;


@end

