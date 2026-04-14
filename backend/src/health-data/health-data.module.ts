import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { CgmReading, CgmReadingSchema } from './schemas/cgm-reading.schema';
import { MoodEntry, MoodEntrySchema } from './schemas/mood-entry.schema';
import { FoodLog, FoodLogSchema } from './schemas/food-log.schema';
import { ActivityLog, ActivityLogSchema } from './schemas/activity-log.schema';
import { HealthDataService } from './health-data.service';
import { CgmController } from './cgm.controller';
import { MoodController } from './mood.controller';
import { FoodController } from './food.controller';
import { AiProxyModule } from '../ai-proxy/ai-proxy.module';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: CgmReading.name, schema: CgmReadingSchema },
      { name: MoodEntry.name, schema: MoodEntrySchema },
      { name: FoodLog.name, schema: FoodLogSchema },
      { name: ActivityLog.name, schema: ActivityLogSchema },
    ]),
    AiProxyModule,
  ],
  controllers: [CgmController, MoodController, FoodController],
  providers: [HealthDataService],
  exports: [HealthDataService],
})
export class HealthDataModule {}
