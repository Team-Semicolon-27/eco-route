import mongoose, {model, Model, models, Schema} from "mongoose";

export interface IDelivery extends Document {
    driverUserId: mongoose.Types.ObjectId;
    managerUserid: mongoose.Types.ObjectId;
    vehicleId: mongoose.Types.ObjectId;
    deadline: Date,
    status: string,
}

export const DeliverySchema = new Schema<IDelivery>({
    driverUserId: { type: mongoose.Schema.Types.ObjectId, required: true, ref: "User" },
    managerUserid: { type: mongoose.Schema.Types.ObjectId, required: true, ref: "User" },
    vehicleId: { type: mongoose.Schema.Types.ObjectId, required: true, ref: "Vehicle" },
    deadline: { type: Date, required: true },
    status: { type: String, enum: ["on way", 'done', "not started"], required: true },
}, { timestamps: true });

export const Delivery: Model<IDelivery> = models.Delivery || model<IDelivery>("Delivery", DeliverySchema);