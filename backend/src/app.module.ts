import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MongooseModule } from '@nestjs/mongoose';
import { ThrottlerModule } from '@nestjs/throttler';
import { HttpModule } from '@nestjs/axios';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { HealthDataModule } from './health-data/health-data.module';
import { ChatModule } from './chat/chat.module';
import { AiProxyModule } from './ai-proxy/ai-proxy.module';
import { User, UserSchema } from './auth/schemas/user.schema';
import { SeedService } from './seed/seed.service';
import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    ThrottlerModule.forRoot([{
      ttl: 60000,
      limit: 10,
    }]),
    MongooseModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        uri: configService.get<string>('MONGODB_URI'),
      }),
      inject: [ConfigService],
    }),
    // Register User model here so SeedService (in AppModule) can inject it
    MongooseModule.forFeature([{ name: User.name, schema: UserSchema }]),
    HttpModule,
    AuthModule,
    UsersModule,
    HealthDataModule,
    ChatModule,
    AiProxyModule,
  ],
  controllers: [AppController],
  providers: [AppService, SeedService],
})
export class AppModule {
  constructor() {
    console.log('='.repeat(60));
    console.log('[AppModule] Initialized');
    console.log('='.repeat(60));
  }
}
